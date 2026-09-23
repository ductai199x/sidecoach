"""Copy onnx-asr's mel preprocessor into the app's assets.

onnx-asr turns a waveform into a NeMo model's mel features with a small ONNX graph of its own
(`onnx_asr/preprocessors/data/nemo80.onnx` or `nemo128.onnx`, as the model's config says) --
not the copy in the model's repository, which differs. The phone runs Parakeet TDT-CTC 110M,
an 80-band model (Tai, 2026-09-23), so it ships `nemo80.onnx`, the one the desktop's onnx-asr
runs for it; this copies it, with the version it came from, whenever onnx-asr changes.

    uv run --no-sync python D:/sidecoach-android/tools/sync_asr.py   (from the desktop repo)
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
from pathlib import Path

import onnx_asr.preprocessors

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets" / "asr"
NAME = "nemo80.onnx"


def main() -> None:
    data = importlib.resources.files(onnx_asr.preprocessors).joinpath("data").joinpath(NAME).read_bytes()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / NAME).write_bytes(data)
    version = importlib.metadata.version("onnx-asr")
    (OUT / "VERSION").write_text(f"onnx-asr {version}\n{NAME} sha256 {hashlib.sha256(data).hexdigest()}\n", encoding="utf-8")
    print(f"{NAME} ({len(data)} bytes), onnx-asr {version} -> {OUT}")


if __name__ == "__main__":
    main()
