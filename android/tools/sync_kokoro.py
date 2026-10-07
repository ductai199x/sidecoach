"""Copy Kokoro's phoneme vocabulary from the desktop's kokoro-onnx into the app's assets.

kokoro-onnx maps each phoneme character to a token id with the vocabulary in its own
`config.json` (`DEFAULT_VOCAB`), because the model files it runs carry none of their own. The
phone reads the same table; this writes it, with the version it came from, whenever the
desktop's kokoro-onnx changes.

    uv run --no-sync python android/tools/sync_kokoro.py   (from the desktop repo)
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from kokoro_onnx.config import DEFAULT_VOCAB

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets" / "kokoro" / "vocab.json"


def main() -> None:
    assert all(len(key) == 1 for key in DEFAULT_VOCAB), "a phoneme of more than one character"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kokoro_onnx": importlib.metadata.version("kokoro-onnx"), "vocab": DEFAULT_VOCAB}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(DEFAULT_VOCAB)} phonemes, kokoro-onnx {payload['kokoro_onnx']} -> {OUT}")


if __name__ == "__main__":
    main()
