"""Film one of the app's pages for the README: a URL at a size, for a while, with scripted clicks
and keys along the way, written as a GIF (and its paints kept for a look). Never shipped;
development only.

    uv run --with playwright --no-sync python tools/media/record_page.py URL OUT.gif \
        --size 720x480 --seconds 8 [--steps steps.json] [--width 640]

A steps file is a list of [seconds, action, argument]: ["click", selector], ["clickxy", "x,y"]
(a point on the page, a board's square), ["press", key], ["type", text], ["hover", selector],
or ["wait", 0]. It drives the installed Microsoft Edge
(Playwright's msedge channel).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

FRAME_MS = 70


def record(url: str, out: Path, size: str, seconds: float, steps: list, width: int) -> None:
    w, h = (int(v) for v in size.split("x"))
    frames: list[tuple[float, bytes]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": w, "height": h})
        page.goto(url)
        page.wait_for_timeout(600)
        cdp = page.context.new_cdp_session(page)

        def on_frame(event: dict) -> None:
            frames.append((event["metadata"]["timestamp"], base64.b64decode(event["data"])))
            cdp.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

        cdp.on("Page.screencastFrame", on_frame)
        cdp.send("Page.startScreencast", {"format": "png", "maxWidth": w, "maxHeight": h})
        started = time.time()
        for at, action, arg in sorted(steps, key=lambda s: s[0]):
            page.wait_for_timeout(max(0, (started + at - time.time()) * 1000))
            if action == "click":
                page.click(arg)
            elif action == "clickxy":
                x, y = (float(v) for v in arg.split(","))
                page.mouse.click(x, y)
            elif action == "press":
                page.keyboard.press(arg)
            elif action == "type":
                page.keyboard.type(arg, delay=50)
            elif action == "hover":
                page.hover(arg)
            if action in ("click", "clickxy"):
                # The pointer leaves what it clicked: left over a page's linked words, it would
                # preview their line on the board (Review's explanations do).
                page.mouse.move(2, h - 2)
        page.wait_for_timeout(max(0, (started + seconds - time.time()) * 1000))
        ended = time.time()
        cdp.send("Page.stopScreencast")
        browser.close()
    # At most a paint every FRAME_MS, but always the one that ends a burst (the page at rest:
    # dropping it would hold a board mid-animation until the next paint).
    kept = [frames[0]]
    for i, (t, png) in enumerate(frames[1:], start=1):
        settles = i + 1 == len(frames) or (frames[i + 1][0] - t) * 1000 >= FRAME_MS
        if (t - kept[-1][0]) * 1000 >= FRAME_MS or settles:
            kept.append((t, png))
    ends = [t for t, _ in kept[1:]] + [ended]
    durations = [max(20, round((e - t) * 1000)) for (t, _), e in zip(kept, ends, strict=True)]
    images = []
    for _, png in kept:
        im = Image.open(io.BytesIO(png)).convert("RGB")
        if width and width != im.width:
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        images.append(im)
    durations[-1] = max(durations[-1], 2500)
    if out.suffix == ".webp":
        # Animated WebP: far smaller than a GIF for a page that moves all over (the splash).
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
            quality=80,
            method=6,
        )
        print(
            f"{out}: {len(images)} frames, {sum(durations) / 1000:.1f} s, "
            f"{out.stat().st_size / 1e6:.1f} MB"
        )
        return
    picks = [images[len(images) * k // 6] for k in range(6)]
    strip = Image.new("RGB", (images[0].width, images[0].height * len(picks)))
    for i, p in enumerate(picks):
        strip.paste(p, (0, i * images[0].height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    quantized[0].save(
        out,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(
        f"{out}: {len(quantized)} frames, {sum(durations) / 1000:.1f} s, "
        f"{out.stat().st_size / 1e6:.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", default="900x720")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--steps", type=Path)
    parser.add_argument("--width", type=int, default=0)
    args = parser.parse_args()
    steps = json.loads(args.steps.read_text(encoding="utf-8")) if args.steps else []
    record(args.url, args.out, args.size, args.seconds, steps, args.width)


if __name__ == "__main__":
    main()
