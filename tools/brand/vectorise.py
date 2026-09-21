"""The chosen logo render, made into the shipped assets: a clean two-colour PNG (hard alpha,
every pixel exactly amber or navy), a traced SVG (vtracer, spline mode, cropped to the
mark with padding), and the square icon source (the mark on a navy rounded tile, 1024 px)
that `npm run tauri icon` turns into the icon set.

    uv run --with vtracer --with resvg-py python tools/brand/vectorise.py \
        data/brand/<stamp>/lens-clean-1.png

Writes beside the repo's brand sources under desktop/brand/: logo.png (the clean mark),
logo.svg, icon-source.png (the tile) and a preview sheet of the SVG rasterised at 16, 32,
64, 256 px (check.png) so the small sizes can be judged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
AMBER = (0xF2, 0xB4, 0x41)
NAVY = (0x14, 0x1A, 0x2A)
# The icon tile is a step lighter than the mark's navy, so the mane stroke stays visible.
TILE = (0x21, 0x2A, 0x45)


def clean(path: Path) -> Image.Image:
    """Hard alpha, and every opaque pixel snapped to the nearer of the two colours."""
    rgba = np.asarray(Image.open(path).convert("RGBA")).astype(np.int32)
    alpha = rgba[:, :, 3]
    # The renders come with a soft halo and no fully opaque pixel: the shape is where the
    # alpha is past half.
    mask = alpha >= 128
    rgb = rgba[:, :, :3]
    d_amber = ((rgb - np.array(AMBER)) ** 2).sum(axis=2)
    d_navy = ((rgb - np.array(NAVY)) ** 2).sum(axis=2)
    out = np.zeros_like(rgba, dtype=np.uint8)
    amber = mask & (d_amber <= d_navy)
    navy = mask & (d_amber > d_navy)
    out[amber] = (*AMBER, 255)
    out[navy] = (*NAVY, 255)
    return Image.fromarray(out, "RGBA")


def bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = np.asarray(image)[:, :, 3]
    ys, xs = np.where(alpha > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def trace(png: Path, svg: Path) -> None:
    import vtracer

    vtracer.convert_image_to_svg_py(
        str(png),
        str(svg),
        colormode="color",
        hierarchical="stacked",
        mode="spline",
        filter_speckle=16,
        color_precision=8,
        layer_difference=48,
        corner_threshold=60,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )


def crop_viewbox(svg: Path, box: tuple[int, int, int, int], pad: int) -> None:
    """A viewBox around the mark with padding, no fixed size: the SVG scales."""
    text = svg.read_text(encoding="utf-8")
    x0, y0, x1, y1 = box
    w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    text = re.sub(r'\s(width|height)="[^"]*"', "", text, count=2)
    text = text.replace("<svg ", f'<svg viewBox="{x0 - pad} {y0 - pad} {w} {h}" ', 1)
    svg.write_text(text, encoding="utf-8")


def rasterise(svg: Path, size: int) -> Image.Image:
    import io

    import resvg_py

    data = resvg_py.svg_to_bytes(svg_path=str(svg), width=size, height=size)
    return Image.open(io.BytesIO(bytes(data))).convert("RGBA")


def tile(mark: Image.Image, size: int = 1024, radius: int = 224, fill: float = 0.78) -> Image.Image:
    """The mark centred on a navy rounded square, the icon source."""
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(out).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=(*TILE, 255)
    )
    x0, y0, x1, y1 = bbox(mark)
    piece = mark.crop((x0, y0, x1, y1))
    scale = fill * size / max(piece.width, piece.height)
    piece = piece.resize((round(piece.width * scale), round(piece.height * scale)), Image.LANCZOS)
    out.alpha_composite(piece, ((size - piece.width) // 2, (size - piece.height) // 2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("render", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "desktop" / "brand")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mark = clean(args.render)
    box = bbox(mark)
    logo_png = args.out / "logo.png"
    mark.crop(box).save(logo_png)
    print("mark", box, "->", logo_png)

    traced_from = args.out / "trace-source.png"
    mark.save(traced_from)
    svg = args.out / "logo.svg"
    trace(traced_from, svg)
    crop_viewbox(svg, box, pad=24)
    traced_from.unlink()
    paths = svg.read_text(encoding="utf-8").count("<path")
    print(f"svg: {paths} paths, {svg.stat().st_size} bytes ->", svg)

    icon = tile(mark)
    icon.save(args.out / "icon-source.png")
    print("icon source ->", args.out / "icon-source.png")

    # The check: the SVG at the sizes an icon lives at, on the app's dark grey and on white.
    sizes = (16, 24, 32, 48, 64, 128, 256)
    sheet = Image.new("RGB", (sum(s + 16 for s in sizes) + 16, 2 * (256 + 24) + 16), "#262421")
    draw = ImageDraw.Draw(sheet)
    draw.rectangle((0, 256 + 32, sheet.width, sheet.height), fill="#ffffff")
    x = 16
    for s in sizes:
        r = rasterise(svg, s)
        sheet.paste(r, (x, 16 + 256 - s), r)
        sheet.paste(r, (x, 256 + 48 + 256 - s), r)
        x += s + 16
    sheet.save(args.out / "check.png")
    print("check ->", args.out / "check.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
