"""Copy onnx-asr's mel preprocessor into the app's assets.

onnx-asr turns a waveform into Parakeet's 128 mel features with a small ONNX graph of its own
(`onnx_asr/preprocessors/data/nemo128.onnx`) -- not the `nemo128.onnx` in the model's
repository, which differs. The phone runs the one the desktop runs; this copies it, with the
version it came from, whenever the desktop's onnx-asr changes.

    uv run --no-sync python D:/sidecoach-android/tools/sync_asr.py   (from the desktop repo)
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
from pathlib import Path

import onnx_asr.preprocessors

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets" / "asr"


def main() -> None:
    data = importlib.resources.files(onnx_asr.preprocessors).joinpath("data").joinpath("nemo128.onnx").read_bytes()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "nemo128.onnx").write_bytes(data)
    version = importlib.metadata.version("onnx-asr")
    (OUT / "VERSION").write_text(f"onnx-asr {version}\nnemo128.onnx sha256 {hashlib.sha256(data).hexdigest()}\n", encoding="utf-8")
    print(f"nemo128.onnx ({len(data)} bytes), onnx-asr {version} -> {OUT}")


if __name__ == "__main__":
    main()
