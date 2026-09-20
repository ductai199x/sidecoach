"""Export the frozen embedding net the square matcher reads cells with (DESIGN.md 5.2).

    uv sync --group classifier
    uv run python tools/classifier/export.py

Ultralytics' YOLO26n-cls -- the smallest of the newest YOLO family, its backbone
pretrained on ImageNet -- cut before its ImageNet linear: the backbone, the Classify
head's 1x1 convolution and its global pool, which give a 1280-wide embedding of a cell
crop. Nothing is trained or fine-tuned (Tai, 2026-09-19): the pieces on screen are named
by cosine similarity of their embeddings to the exemplars taken from the start
position, in `sidecoach.vision.embedding`. The graph takes BGR uint8 pixels as floats
(0..255) at EMBED_INPUT px square, the channel flip and the scaling inside, and returns
the unit-normalised embedding.

The weights are fetched from Ultralytics' releases (AGPL-3.0) on first use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.vision.embedding import EMBED_INPUT, MODEL_PATH  # noqa: E402

WEIGHTS = ROOT / "data/classifier/yolo26n-cls.pt"


class Embedder(nn.Module):
    def __init__(self, weights: Path) -> None:
        super().__init__()
        from ultralytics import YOLO
        from ultralytics.utils.downloads import attempt_download_asset

        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            attempt_download_asset(str(weights))
        layers = list(YOLO(str(weights)).model.model)
        classify = layers[-1]
        self.backbone = nn.Sequential(*layers[:-1])
        self.conv = classify.conv
        self.pool = classify.pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flip(1) / 255.0  # BGR 0..255 -> RGB 0..1, as the net was trained
        v = self.pool(self.conv(self.backbone(x))).flatten(1)
        return v / (v.norm(dim=1, keepdim=True) + 1e-6)


def main() -> None:
    model = Embedder(WEIGHTS).eval()
    example = torch.zeros(1, 3, EMBED_INPUT, EMBED_INPUT)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(MODEL_PATH),
        input_names=["cells"],
        output_names=["embedding"],
        dynamic_axes={"cells": {0: "n"}, "embedding": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    # The export must reproduce the torch net.
    import onnxruntime as ort

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    x = (
        np.random.default_rng(0)
        .integers(0, 256, (3, 3, EMBED_INPUT, EMBED_INPUT))
        .astype(np.float32)
    )
    with torch.no_grad():
        want = model(torch.from_numpy(x)).numpy()
    got = session.run(None, {"cells": x})[0]
    gap = float(np.abs(want - got).max())
    size = MODEL_PATH.stat().st_size // 1024
    print(f"exported {MODEL_PATH} ({size} KB); torch vs onnx gap {gap:.2e}")
    assert gap < 1e-4


if __name__ == "__main__":
    main()
