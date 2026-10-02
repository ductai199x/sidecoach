"""Cell embeddings from a frozen pretrained net, for naming pieces by similarity.

Tai's design (2026-09-19): no training, no fine-tuning. The pieces on screen are named
by comparing each cell's embedding with the embeddings of the cells of the start
position, where every piece kind and both empty square colours are on view: the
nearest of those 64 exemplars, by cosine similarity, names the cell. The net is
Ultralytics' YOLO26n-cls, the smallest of the newest YOLO family, pretrained on ImageNet
and cut before its ImageNet linear (tools/classifier/export.py): its 1280-wide pooled
embedding, unit-normalised, comes out of `models/yolo26n-embed.onnx`. On twelve piece
sets, three themes and three cell sizes, with a highlight, a check glow, a badge and an
arrow on the boards, it named 99.8% of cells right at 224 px input, 99.6% at 128 or
160 (the rest all but a few under the arrow, which the drawing rule turns unknown), the
silhouette overlap's failure cases (a shaded set on a board of its own tone, thin
detail at 44 px) among them.

Only what the cell IS comes from the embedding. Whether a piece has landed on its
square is the phase correlation of the cell against that piece's own start-position
crop (`pieces._phase_shift`: the shift itself, sub-pixel, no silhouette -- a pooled
embedding barely moves when the piece does, 0.87 similar at a fifth of a cell off, and
its shifted copies were no better at 43 px), and a tint, a hover outline, a cover or a
drawing over the cell are the colour and geometry rules in `pieces`, which need no
training and no theme. Runs through onnxruntime on the GPU when there is one, else the
CPU, the changed cells of a frame in one batch. The model file ships in the package;
`CellEmbedder.load()` is None without it (a broken install), and nothing is learned.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).with_name("models") / "yolo26n-embed.onnx"
# Where the net runs: "auto" takes the GPU when onnxruntime's CUDA provider initialises,
# "cuda" insists, "cpu" never touches the GPU (see voice.runtime, which chooses the same
# way for the voice models). A 3090 embeds 64 cells at 224 px in a few milliseconds.
DEVICE = os.environ.get("SIDECOACH_EMBED_DEVICE", "auto")
# The cell crop (2 x squares.CELL = 96 px) is scaled up to this before the net: the net
# was trained at 224, and the read is best there (0.22% wrong at 224, 0.43% at 128 and
# 160, 2% at 96). The cost grows with the area; accuracy comes first (Tai).
EMBED_INPUT = 224
# An exemplar this similar names the cell; below it nothing on the start position looks
# like what is there (a label on an empty square, a dialog's edge, a piece under an arrow).
MIN_SIMILARITY = 0.55
# The lead the nearest exemplar's kind needs over the nearest of another kind.
MIN_MARGIN = 0.03
EMBED_DIM = 1280
# Embeddings kept by the crop's content (5 kB each): a few boards' worth of pieces.
CACHED = 2048


class CellEmbedder:
    """The net, loaded once; `embed` over a batch of cell crops."""

    def __init__(self, path: Path = MODEL_PATH, device: str = DEVICE) -> None:
        from sidecoach.voice.runtime import session

        # The GPU when there is one (the voice models' runtime picks it the same way);
        # the CPU otherwise, with two threads, the engine wanting the rest.
        self.session, self.device = session(path, device, intra_op_num_threads=2)
        self.path = path
        # Embeddings by the crop's bytes: a piece that lands on a square was seen there, or
        # on another square of its colour, a moment ago -- the same pixels, the same vector,
        # and no run of the net for it. The newest CACHED are kept.
        self._seen: OrderedDict[bytes, np.ndarray] = OrderedDict()
        self._seen_lock = threading.Lock()

    @staticmethod
    @lru_cache(maxsize=1)
    def load(path: Path = MODEL_PATH) -> CellEmbedder | None:
        """The embedder, or None when there is no model file to load."""
        if not path.exists():
            return None
        return CellEmbedder(path)

    @staticmethod
    def prepare(crop: np.ndarray) -> np.ndarray:
        """A cell crop (BGR uint8) at the net's input size."""
        if crop.shape[0] == EMBED_INPUT and crop.shape[1] == EMBED_INPUT:
            return crop
        growing = crop.shape[0] < EMBED_INPUT
        return cv2.resize(
            crop,
            (EMBED_INPUT, EMBED_INPUT),
            interpolation=cv2.INTER_CUBIC if growing else cv2.INTER_AREA,
        )

    def prime(self) -> float:
        """One start-position-sized batch through the net, so the first real one does not
        pay the runtime's warm-up (a second on CUDA). Returns the seconds it took."""
        begun = time.perf_counter()
        self.embed([np.zeros((EMBED_INPUT, EMBED_INPUT, 3), dtype=np.uint8)] * 64)
        return time.perf_counter() - begun

    def _run(self, batch: np.ndarray) -> np.ndarray:
        """The net over a batch. A run that fails off the CPU -- CUDA out of memory, a
        driver reset -- loads the net again on the CPU, for good, and runs there: the GPU
        session was kept failing every frame, and "pipeline error" filled the log (round 2,
        finding 5). A failure on the CPU is raised."""
        try:
            run = self.session.run(None, {"cells": batch})
        except Exception:
            if self.device.name == "cpu":
                raise
            log.warning(
                "the embedding net failed on %s; on the CPU from now on",
                self.device.name,
                exc_info=True,
            )
            from sidecoach.voice.runtime import session

            self.session, self.device = session(self.path, "cpu", intra_op_num_threads=2)
            run = self.session.run(None, {"cells": batch})
        return np.asarray(run[0], dtype=np.float32)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Unit embeddings (n, EMBED_DIM), float32, one per crop; a crop seen before is not
        run through the net again."""
        if not crops:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        keys = [
            hashlib.blake2b(
                np.ascontiguousarray(c).tobytes() + str(c.shape).encode(), digest_size=16
            ).digest()
            for c in crops
        ]
        out = np.empty((len(crops), EMBED_DIM), dtype=np.float32)
        todo: list[int] = []
        with self._seen_lock:
            for i, key in enumerate(keys):
                hit = self._seen.get(key)
                if hit is None:
                    todo.append(i)
                else:
                    self._seen.move_to_end(key)
                    out[i] = hit
        if todo:
            batch = np.stack([self.prepare(crops[i]) for i in todo]).transpose(0, 3, 1, 2)
            fresh = self._run(np.ascontiguousarray(batch, dtype=np.float32))
            with self._seen_lock:
                for j, i in enumerate(todo):
                    out[i] = fresh[j]
                    self._seen[keys[i]] = fresh[j]
                while len(self._seen) > CACHED:
                    self._seen.popitem(last=False)
        return out
