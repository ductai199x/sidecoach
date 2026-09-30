"""Sidecoach's piece set as SVG, from the round-5 sheet (round5-sheet.png, README.md): each
piece found by its own shape, split into its flat colour layers (the silhouette in the
outline colour, the body in its main tone, the shadow band), each layer smoothed and traced
with potrace, the layers stacked. All 12 at one scale, bases on one line, in a square
viewBox. Then each SVG rendered and subtracted from its source: the silhouettes' overlap
and the colour error, per piece.

    uv run --no-project --with pillow --with numpy --with opencv-python-headless \
        --with potracer --with resvg-py python tools/pieces/to_svg.py [out-dir]

Writes desktop/static/pieces/{wK..bP}.svg unless another folder is named."""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
import potrace
import resvg_py
from PIL import Image

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "desktop" / "static" / "pieces"
ORDER = ["K", "Q", "R", "B", "N", "P"]
UP = 8  # the native crop is traced at 8x: smooth masks, then potrace's curves
VIEW = 512  # the viewBox side
FILL, BOTTOM = 0.86, 0.07  # the tallest piece's share of the square; the base's margin
PAD = 4

# The sheet's own tones (k-means over each side's pixels), one outline, main and shadow a side.
PALETTE = {
    "w": {"outline": "#2A221E", "main": "#F6F0E9", "shadow": "#D4CAC0"},
    "b": {"outline": "#27201C", "main": "#655C56", "shadow": "#49423D"},
}


def rgb(h: str) -> np.ndarray:
    return np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


def trace(mask: np.ndarray) -> str:
    """A mask (at UP x) as SVG path data in native pixels, potrace's smooth curves."""
    bm = potrace.Bitmap(~mask.astype(bool))  # potracer traces the pixels that are NOT set
    path = bm.trace(turdsize=int(4 * UP * UP), turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY, alphamax=1.0, opticurve=True, opttolerance=0.4)
    f = lambda p: f"{p.x / UP:.2f},{p.y / UP:.2f}"  # noqa: E731
    out = []
    for curve in path:
        out.append(f"M{f(curve.start_point)}")
        for seg in curve.segments:
            out.append(f"L{f(seg.c)}L{f(seg.end_point)}" if seg.is_corner else f"C{f(seg.c1)} {f(seg.c2)} {f(seg.end_point)}")
        out.append("Z")
    return "".join(out)


def smooth(mask: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) > 0.5


sheet = np.array(Image.open(HERE / "round5-sheet.png").convert("RGBA"))
solid = (sheet[..., 3] > 200).astype(np.uint8)
solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
joined = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
n, _, stats, _ = cv2.connectedComponentsWithStats(joined)
found = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:12]
boxes = [tuple(int(v) for v in stats[i, :4]) for i in found]
top = sorted(boxes, key=lambda b: b[1] + b[3])[:6]  # the white row stands higher
bottom = [b for b in boxes if b not in top]
scale = min(FILL * VIEW / max(b[3] for b in boxes), 0.92 * VIEW / max(b[2] for b in boxes))

OUT.mkdir(parents=True, exist_ok=True)
results = []
for colour, row in (("w", top), ("b", bottom)):
    pal = PALETTE[colour]
    centres = np.stack([rgb(pal[k]) for k in ("outline", "main", "shadow")])
    for p, (x, y, w, h) in zip(ORDER, sorted(row, key=lambda b: b[0]), strict=True):
        x0, y0 = max(0, x - PAD), max(0, y - PAD)
        crop = sheet[y0 : y + h + PAD, x0 : x + w + PAD]
        big = cv2.resize(crop, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        body = big[..., 3] > 127
        near = ((big[..., None, :3] - centres[None, None]) ** 2).sum(-1).argmin(-1)
        silhouette = smooth(body, UP * 0.6)
        light = smooth(body & (near != 0), UP * 0.7) & silhouette
        shadow = smooth(body & (near == 2), UP * 0.9) & light
        layers = [("outline", silhouette), ("main", light), ("shadow", shadow)]
        sw, sh = round(crop.shape[1] * scale), round(crop.shape[0] * scale)
        ox, oy = (VIEW - sw) // 2, round(VIEW * (1 - BOTTOM)) - sh
        paths = "".join(f'<path fill="{pal[k]}" fill-rule="evenodd" d="{trace(m)}"/>' for k, m in layers if m.any())
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW} {VIEW}"><g transform="translate({ox} {oy}) scale({scale:.5f})">{paths}</g></svg>'
        (OUT / f"{colour}{p}.svg").write_text(svg, encoding="utf-8")
        # The check: the source crop placed as the SVG places it, against the SVG rendered.
        ref = np.zeros((VIEW, VIEW, 4), np.float32)
        placed = np.array(Image.fromarray(crop).resize((sw, sh), Image.LANCZOS)).astype(np.float32)
        ref[oy : oy + sh, ox : ox + sw] = placed
        got = np.array(Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, width=VIEW, height=VIEW)))).convert("RGBA")).astype(np.float32)
        a, b = ref[..., 3] > 127, got[..., 3] > 127
        iou = (a & b).sum() / max(1, (a | b).sum())
        err = np.abs(ref[..., :3] - got[..., :3])[a & b].mean()
        results.append((colour + p, iou, err, len(svg)))

for c, iou, err, size in results:
    print(f"{c}  silhouette overlap {iou:.4f}  colour error {err:5.2f}/255  {size} bytes")
print(f"mean overlap {np.mean([r[1] for r in results]):.4f}, mean colour error {np.mean([r[2] for r in results]):.2f} -> {OUT}")
