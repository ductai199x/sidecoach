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
crop (`pieces.LANDING`, "phase": the shift itself, sub-pixel, no silhouette -- a pooled
embedding barely moves when the piece does, 0.87 similar at a fifth of a cell off, and
its shifted copies were no better at 43 px), and a tint, a hover outline, a cover or a
drawing over the cell are the colour and geometry rules in `pieces`, which need no
training and no theme. Runs through onnxruntime on the GPU when there is one, else the
CPU, the changed cells of a frame in one batch; `CellEmbedder.load()` is None without
the model file, and the overlap reads on.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

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


class CellEmbedder:
    """The net, loaded once; `embed` over a batch of cell crops."""

    def __init__(self, path: Path = MODEL_PATH, device: str = DEVICE) -> None:
        from sidecoach.voice.runtime import session

        # The GPU when there is one (the voice models' runtime picks it the same way);
        # the CPU otherwise, with two threads, the engine wanting the rest.
        self.session, self.device = session(path, device, intra_op_num_threads=2)
        self.path = path

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

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Unit embeddings (n, EMBED_DIM), float32, one per crop."""
        if not crops:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        batch = np.stack([self.prepare(c) for c in crops]).transpose(0, 3, 1, 2)
        out = self.session.run(None, {"cells": np.ascontiguousarray(batch, dtype=np.float32)})
        return np.asarray(out[0], dtype=np.float32)
