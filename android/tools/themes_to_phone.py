"""The desktop's learned sets (theme-*.pkl, cache version 13) written as the phone's theme files
(version 6, `ThemeStore.write`): DataOutputStream, big-endian, field for field."""

import pickle
import struct
import sys
from pathlib import Path

import numpy as np


def utf(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack(">H", len(raw)) + raw


def bits(a: np.ndarray) -> bytes:
    flat = np.asarray(a, dtype=bool).ravel()
    out = bytearray(struct.pack(">i", flat.size))
    packed = np.packbits(flat, bitorder="little")  # bit i of byte i//8 is flat[i], as writeBits
    out += packed.tobytes()
    return bytes(out)


def doubles(values) -> bytes:
    return b"".join(struct.pack(">d", float(v)) for v in values)


def convert(model) -> bytes:
    out = bytearray(struct.pack(">i", 6))
    out += doubles(model.light_bg) + doubles(model.dark_bg)
    out += struct.pack(">?", True)  # whiteAtBottom: the phone reads orientation from the labels
    out += struct.pack(">i", len(model.exemplars))
    for e in model.exemplars:
        out += utf(e.symbol) + bits(e.tone) + struct.pack(">d", float(e.brightness))
    out += struct.pack(">i", len(model.piece_colours))
    for fill, edge in model.piece_colours:
        out += doubles(fill) + struct.pack(">?", edge is not None)
        if edge is not None:
            out += doubles(edge)
    out += struct.pack(">i", len(model.palette))
    for colour in model.palette:
        out += doubles(colour)
    vectors = np.asarray(model.exemplar_vectors, dtype=np.float16)
    out += struct.pack(">ii", vectors.shape[0], vectors.shape[1])
    out += vectors.view(np.uint16).astype(">u2").tobytes()
    out += struct.pack(">i", len(model.exemplar_labels))
    for label in model.exemplar_labels:
        out += utf(label)
    out += struct.pack(">i", len(model.rank_labels))
    for rank, glyph in model.rank_labels.items():
        out += struct.pack(">i", int(rank)) + bits(glyph)
    out += struct.pack(">i", len(model.landing_crops))
    for (symbol, light), crop in model.landing_crops.items():
        crop = np.asarray(crop, dtype=np.uint8)
        out += utf(symbol) + struct.pack(">?ii", bool(light), crop.shape[0], crop.shape[1])
        out += crop.tobytes()
    out += struct.pack(">d", float(model.landing_px))
    return bytes(out)


src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for path in sorted(src.glob("theme-*.pkl")):
    payload = pickle.load(path.open("rb"))
    target = dst / (path.stem + ".bin")
    target.write_bytes(convert(payload["model"]))
    print(target.name, target.stat().st_size)
