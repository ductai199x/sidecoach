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
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

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
# The one batch size the net runs at on the GPU (`CellEmbedder._batched`): a move's few new
# cells in one run of about 3 ms on a 3090, a start position's 64 in four.
GPU_BATCH = 16
# Embeddings kept by the crop's content (5 kB each): a few boards' worth of pieces.
CACHED = 2048
# After a run fails on the GPU the net runs on the CPU, and the GPU is tried again this long
# after, off the frame's path; back on it if it works, else again as long after (Tai,
# 2026-10-02, round 3 audit E9: one driver reset kept it on the CPU until a restart).
GPU_RETRY_S = 60.0


def _in_background(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="sidecoach-embed-gpu", daemon=True).start()


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
        # The session and its device are swapped together, under this lock.
        self._swap = threading.Lock()
        # On the CPU after a run failed on the GPU: the device it failed on, when it is tried
        # again (the clock of `time.monotonic`), and whether a try is under way.
        self._lost: str | None = None
        self._retry_at = 0.0
        self._retrying = False
        self._cpu: tuple[object, object] | None = None  # the CPU session, loaded once
        # Where the GPU is tried again: a thread of its own; a test runs it in place.
        self.spawn: Callable[[Callable[[], None]], None] = _in_background

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
        """One batch through the net at the size it runs at, so the first real one does not
        pay the runtime's warm-up (a second on CUDA). Returns the seconds it took."""
        begun = time.perf_counter()
        size = GPU_BATCH if self.device.name != "cpu" else 64
        self._run(np.zeros((size, 3, EMBED_INPUT, EMBED_INPUT), dtype=np.float32))
        return time.perf_counter() - begun

    def _batched(self, batch: np.ndarray) -> np.ndarray:
        """The net over a batch of any size. On the GPU every change of batch size cost 30
        to 70 ms, at whatever point in a session, whatever cuDNN's search was set to, against
        2 to 8 ms for a run at the size before; and with the crop cache a frame's batch is
        whatever count of cells is new (round 2, finding 8). There the net runs only at
        GPU_BATCH: in pieces of it, the last padded. The CPU runs the batch as it is."""
        if self.device.name == "cpu":
            return self._run(batch)
        out: list[np.ndarray] = []
        for start in range(0, len(batch), GPU_BATCH):
            piece = batch[start : start + GPU_BATCH]
            n = len(piece)
            if n < GPU_BATCH:
                pad = np.zeros((GPU_BATCH - n, *piece.shape[1:]), dtype=piece.dtype)
                piece = np.concatenate([piece, pad])
            out.append(self._run(piece)[:n])
        return np.concatenate(out)

    def _run(self, batch: np.ndarray) -> np.ndarray:
        """The net over a batch. A run that fails off the CPU -- CUDA out of memory, a
        driver reset -- loads the net on the CPU and runs there: the GPU session was kept
        failing every frame, and "pipeline error" filled the log (round 2, finding 5). The GPU
        is tried again GPU_RETRY_S later, off the frame's path (`_retry_gpu`), never at every
        frame. A failure on the CPU is raised."""
        self._retry_due()
        with self._swap:
            sess, device = self.session, self.device
        try:
            run = sess.run(None, {"cells": batch})
        except Exception:
            if device.name == "cpu":
                raise
            log.warning(
                "the embedding net failed on %s; on the CPU, the GPU tried again in %.0f s",
                device.name,
                GPU_RETRY_S,
                exc_info=True,
            )
            sess = self._to_cpu(device.name)
            run = sess.run(None, {"cells": batch})
        return np.asarray(run[0], dtype=np.float32)

    def _to_cpu(self, lost: str) -> Any:
        """The net on the CPU, after a run failed on `lost`; the GPU tried again later."""
        if self._cpu is None:
            from sidecoach.voice.runtime import session

            self._cpu = session(self.path, "cpu", intra_op_num_threads=2)
        with self._swap:
            self.session, self.device = self._cpu
            self._lost = lost
            self._retry_at = time.monotonic() + GPU_RETRY_S
            return self.session

    def _retry_due(self) -> None:
        """On the CPU since the GPU failed, and GPU_RETRY_S gone by: the GPU tried again, on
        its own thread; the frame goes on on the CPU meanwhile."""
        with self._swap:
            if self._lost is None or self._retrying or time.monotonic() < self._retry_at:
                return
            self._retrying = True
            lost = self._lost
        self.spawn(lambda: self._retry_gpu(lost))

    def _retry_gpu(self, device: str) -> None:
        """The net loaded on `device` again and run once at the size it runs at there (which
        proves it, and primes it): the GPU's from then on if it works; else the CPU's for
        another GPU_RETRY_S."""
        try:
            from sidecoach.voice.runtime import session

            sess, where = session(self.path, device, intra_op_num_threads=2)
            if where.name == "cpu":
                raise RuntimeError(where.detail)
            sess.run(
                None, {"cells": np.zeros((GPU_BATCH, 3, EMBED_INPUT, EMBED_INPUT), np.float32)}
            )
        except Exception as exc:  # noqa: BLE001 - still failing: the CPU, and another try later
            log.info(
                "the embedding net still fails on %s (%s); tried again in %.0f s",
                device,
                exc,
                GPU_RETRY_S,
            )
            with self._swap:
                self._retry_at = time.monotonic() + GPU_RETRY_S
                self._retrying = False
            return
        with self._swap:
            self.session, self.device = sess, where
            self._lost = None
            self._retrying = False
        log.warning("the embedding net is back on %s", where.name)

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
            fresh = self._batched(np.ascontiguousarray(batch, dtype=np.float32))
            with self._seen_lock:
                for j, i in enumerate(todo):
                    out[i] = fresh[j]
                    self._seen[keys[i]] = fresh[j]
                while len(self._seen) > CACHED:
                    self._seen.popitem(last=False)
        return out
