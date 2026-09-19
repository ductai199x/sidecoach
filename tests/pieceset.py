"""A real piece set for the renderer: the cburnett pieces, rasterised from their vectors.

The polygons in `tests/synth.py` stand in for the simplest sets; real sets are thin
strokes, crowns with points, a knight's mane, a bishop's slit, detail lines in the other
side's colour inside every piece -- the shapes the recogniser meets on a live board.
python-chess carries the vectors of Colin M.L. Burnett's set (lichess's default; GPLv2+,
BSD and GFDL) in `chess.svg.PIECES`, a 45-unit square per piece, and this module
rasterises them the way a browser does, at the size of the cell, with anti-aliased edges:
paths (lines, cubic and smooth curves, arcs), circles, a transform, the even-odd fill
rule, strokes of the set's width. The colours can be swapped, which is how a wooden or a
coloured variant of the set is drawn.

Sprites are rendered once per colouring at `SPRITE` pixels, premultiplied, and scaled to
each cell with area interpolation, which is what a vector rasterised at the cell's size
looks like to within the anti-aliasing.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from functools import cache

import chess.svg
import cv2
import numpy as np

SPRITE = 512
UNITS = 45.0  # the pieces' viewBox
_SUPER = 2  # rendered at twice the sprite size, then averaged down
_NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_TOKEN = re.compile(r"[MmLlHhVvCcSsAaZz]|" + _NUMBER.pattern)
_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "A": 7, "Z": 0}

# The set's own colours, and how a variant swaps them.
Colouring = dict[str, str]
WHITE_SET: Colouring = {}
# Cream pieces with brown outlines against dark-brown pieces with tan detail.
WOODEN_SET: Colouring = {
    "#fff": "#f5e1be",
    "#ffffff": "#f5e1be",
    "#000": "#5a462d",
    "#000000": "#5a462d",
    "#ececec": "#d7bc8f",
}


def _colour(spec: str, colouring: Colouring) -> tuple[int, int, int] | None:
    """BGR for an SVG colour, None for `none`."""
    spec = colouring.get(spec.strip().lower(), spec.strip().lower())
    if spec == "none":
        return None
    if spec.startswith("#"):
        hexes = spec[1:]
        if len(hexes) == 3:
            hexes = "".join(ch * 2 for ch in hexes)
        r, g, b = (int(hexes[i : i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)
    raise ValueError(f"unsupported colour {spec!r}")


def _style(element: ET.Element, inherited: dict[str, str]) -> dict[str, str]:
    """Presentation attributes, the `style` attribute on top, over what the group set."""
    style = dict(inherited)
    for key, value in element.attrib.items():
        if key in ("fill", "stroke", "stroke-width", "fill-rule"):
            style[key] = value
    for part in element.attrib.get("style", "").split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            style[key.strip()] = value.strip()
    return style


def _bezier(p0, p1, p2, p3, steps: int = 24) -> list[tuple[float, float]]:
    out = []
    for i in range(1, steps + 1):
        t = i / steps
        u = 1 - t
        out.append(
            (
                u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1],
            )
        )
    return out


def _arc(p0, rx, ry, rotation, large, sweep, p1, steps: int = 24) -> list[tuple[float, float]]:
    """An SVG elliptical arc from p0 to p1 as points (the endpoint-to-centre conversion of
    the SVG implementation notes, F.6.5)."""
    if rx == 0 or ry == 0 or p0 == p1:
        return [p1]
    phi = math.radians(rotation)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2, (p0[1] - p1[1]) / 2
    x1 = cos_phi * dx + sin_phi * dy
    y1 = -sin_phi * dx + cos_phi * dy
    rx, ry = abs(rx), abs(ry)
    scale = x1 * x1 / (rx * rx) + y1 * y1 / (ry * ry)
    if scale > 1:
        rx *= math.sqrt(scale)
        ry *= math.sqrt(scale)
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    factor = math.sqrt(max(0.0, num / den)) if den else 0.0
    if large == sweep:
        factor = -factor
    cx1 = factor * rx * y1 / ry
    cy1 = -factor * ry * x1 / rx
    cx = cos_phi * cx1 - sin_phi * cy1 + (p0[0] + p1[0]) / 2
    cy = sin_phi * cx1 + cos_phi * cy1 + (p0[1] + p1[1]) / 2

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / length)))
        return -a if ux * vy - uy * vx < 0 else a

    start = angle(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    delta = angle((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi
    out = []
    for i in range(1, steps + 1):
        theta = start + delta * i / steps
        x = rx * math.cos(theta)
        y = ry * math.sin(theta)
        out.append((cos_phi * x - sin_phi * y + cx, sin_phi * x + cos_phi * y + cy))
    out[-1] = p1
    return out


def _subpaths(d: str) -> list[tuple[list[tuple[float, float]], bool]]:
    """The polylines of a path, each with whether it was closed."""
    tokens = _TOKEN.findall(d)
    out: list[tuple[list[tuple[float, float]], bool]] = []
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    last_control: tuple[float, float] | None = None
    command = ""
    i = 0

    def flush(closed: bool) -> None:
        nonlocal points
        if len(points) > 1:
            out.append((points, closed))
        points = []

    while i < len(tokens):
        token = tokens[i]
        if token.isalpha():
            command = token
            i += 1
            if command in "Zz":
                flush(True)
                current = start
                points = [current]
                last_control = None
                continue
        upper = command.upper()
        n = _ARGS[upper]
        args = [float(t) for t in tokens[i : i + n]]
        i += n
        relative = command.islower()
        ox, oy = current if relative else (0.0, 0.0)
        if upper == "M":
            flush(False)
            current = (ox + args[0], oy + args[1])
            start = current
            points = [current]
            command = "l" if relative else "L"  # further pairs are lines
            last_control = None
        elif upper == "L":
            current = (ox + args[0], oy + args[1])
            points.append(current)
            last_control = None
        elif upper == "H":
            current = (ox + args[0], current[1])
            points.append(current)
            last_control = None
        elif upper == "V":
            current = (current[0], oy + args[0])
            points.append(current)
            last_control = None
        elif upper in ("C", "S"):
            if upper == "C":
                c1 = (ox + args[0], oy + args[1])
                c2 = (ox + args[2], oy + args[3])
                end = (ox + args[4], oy + args[5])
            else:
                c1 = (
                    (2 * current[0] - last_control[0], 2 * current[1] - last_control[1])
                    if last_control
                    else current
                )
                c2 = (ox + args[0], oy + args[1])
                end = (ox + args[2], oy + args[3])
            points += _bezier(current, c1, c2, end)
            last_control = c2
            current = end
        elif upper == "A":
            end = (ox + args[5], oy + args[6])
            points += _arc(current, args[0], args[1], args[2], bool(args[3]), bool(args[4]), end)
            current = end
            last_control = None
    flush(False)
    return out


def _transform(spec: str | None):
    """A point transform for `matrix(a,b,c,d,e,f)`, or the identity."""
    if not spec:
        return lambda p: p
    match = re.fullmatch(r"\s*matrix\(([^)]*)\)\s*", spec)
    if not match:
        raise ValueError(f"unsupported transform {spec!r}")
    a, b, c, d, e, f = (float(x) for x in re.split(r"[\s,]+", match.group(1).strip()))
    return lambda p: (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)


def _shapes(
    symbol: str,
) -> list[tuple[list[tuple[list[tuple[float, float]], bool]], dict[str, str]]]:
    """Every shape of a piece: its subpaths in viewBox units and its style."""
    root = ET.fromstring(chess.svg.PIECES[symbol])
    shapes: list[tuple[list[tuple[list[tuple[float, float]], bool]], dict[str, str]]] = []

    def walk(group: ET.Element, inherited: dict[str, str]) -> None:
        for element in group:
            tag = element.tag.split("}")[-1]
            style = _style(element, inherited)
            if tag == "g":
                walk(element, style)
                continue
            transform = _transform(element.attrib.get("transform"))
            if tag == "path":
                subpaths = [
                    ([transform(p) for p in pts], closed)
                    for pts, closed in _subpaths(element.attrib["d"])
                ]
            elif tag == "circle":
                cx, cy, r = (float(element.attrib[k]) for k in ("cx", "cy", "r"))
                ring = [
                    transform((cx + r * math.cos(t), cy + r * math.sin(t)))
                    for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)
                ]
                subpaths = [(ring, True)]
            else:
                continue
            shapes.append((subpaths, style))

    walk(root, _style(root, {"fill": "#000", "stroke": "none", "stroke-width": "1"}))
    return shapes


def _coverage(draw) -> np.ndarray:
    """Anti-aliased coverage of what `draw` paints, 0 to 1, at sprite size."""
    canvas = np.zeros((SPRITE * _SUPER, SPRITE * _SUPER), dtype=np.uint8)
    draw(canvas)
    return (
        cv2.resize(canvas, (SPRITE, SPRITE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    )


def _render(symbol: str, colouring: Colouring) -> tuple[np.ndarray, np.ndarray]:
    """A piece as premultiplied BGR and alpha, both float32 at sprite size."""
    scale = SPRITE * _SUPER / UNITS
    rgb = np.zeros((SPRITE, SPRITE, 3), dtype=np.float32)
    alpha = np.zeros((SPRITE, SPRITE), dtype=np.float32)

    def over(coverage: np.ndarray, colour: tuple[int, int, int]) -> None:
        nonlocal rgb, alpha
        c = coverage[..., None]
        rgb = np.array(colour, dtype=np.float32) * c + rgb * (1 - c)
        alpha = coverage + alpha * (1 - coverage)

    for subpaths, style in _shapes(symbol):
        polys = [
            np.round(np.array(pts, dtype=np.float64) * scale).astype(np.int32)
            for pts, _ in subpaths
        ]
        fill = _colour(style.get("fill", "none"), colouring)
        stroke = _colour(style.get("stroke", "none"), colouring)
        width = float(style.get("stroke-width", "1")) * scale
        if fill is not None:
            over(
                _coverage(
                    lambda cv, polys=polys: cv2.fillPoly(cv, polys, 255, lineType=cv2.LINE_AA)
                ),
                fill,
            )
        if stroke is not None and width > 0:
            closed = [is_closed for _, is_closed in subpaths]

            def strokes(cv, polys=polys, closed=tuple(closed), width=width):
                for poly, is_closed in zip(polys, closed, strict=True):
                    cv2.polylines(
                        cv, [poly], is_closed, 255, max(1, int(round(width))), lineType=cv2.LINE_AA
                    )

            over(_coverage(strokes), stroke)
    return rgb, alpha


@cache
def _sprites(
    colouring_key: tuple[tuple[str, str], ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    colouring = dict(colouring_key)
    return {symbol: _render(symbol, colouring) for symbol in chess.svg.PIECES}


def sprite(symbol: str, colouring: Colouring | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The premultiplied BGR and alpha of a piece at sprite size."""
    return _sprites(tuple(sorted((colouring or {}).items())))[symbol]


def draw_sprite(
    frame: np.ndarray,
    x: int,
    y: int,
    size: int,
    symbol: str,
    colouring: Colouring | None = None,
) -> None:
    """Composite a piece, scaled to `size` pixels square, with its top-left corner at
    (x, y), clipped to the frame."""
    rgb, alpha = sprite(symbol, colouring)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + size), min(h, y + size)
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - x, y0 - y
    rgb = rgb[sy : sy + (y1 - y0), sx : sx + (x1 - x0)]
    a = alpha[sy : sy + (y1 - y0), sx : sx + (x1 - x0), None]
    patch = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(rgb + patch * (1 - a), 0, 255).astype(np.uint8)
