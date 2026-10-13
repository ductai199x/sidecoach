"""The README's live GIFs: the filmed site (film_site.py) and the filmed panel (record_live.py)
set side by side as two windows, the idle stretches sped up, and a line at the bottom saying
what is happening. Never shipped; development only.

    uv run --no-sync python tools/media/compose.py REC CAPTURE OUT.gif --story hero

STORIES below say which line shows from which moment (a ply reached, or a line the coach said).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path("C:/Windows/Fonts")
BACKDROP = (17, 19, 24)
WINDOW_BAR = (38, 40, 46)
BAR_TEXT = (170, 172, 178)
CAPTION = (236, 237, 240)
ACCENT = (232, 176, 64)
GAP = 22
BAR_H = 26
FPS = 10
FAST = 5.0  # how much faster the idle stretches play


@dataclass(frozen=True)
class Beat:
    kind: str  # "start", "ply" (of the first game), "said", or "game" (the nth game, from 1)
    key: object
    line: str
    # Seconds after the moment (a ply of a later game: "game" with a delay)
    after: float = 0.0


@dataclass(frozen=True)
class Story:
    beats: list[Beat]
    # Moments before this ply play fast throughout (an opening nobody needs to watch).
    full_from: int = 0
    # How fast the moments themselves play
    pace: float = 1.0
    # The film starts a little before this ply of the first game (0: when the game starts)
    start_ply: int = 0


STORIES: dict[str, Story] = {
    "looks": Story(
        pace=2.0,
        beats=[
            Beat("start", None, "Any board: its colours, its pieces, its size."),
            Beat("game", 1, "The window resized mid-game? It keeps up.", after=9.0),
            Beat("game", 2, "Playing Black? It reads the board the right way round."),
            Beat("game", 3, "New pieces and colours: learned from the first position."),
            Beat("game", 4, "No extension, no site access: only the pixels on your screen."),
        ],
    ),
    # The AI coach on the same game (record_live.py --actions ask): a question, the two hints.
    "hero_ai": Story(
        full_from=6,
        start_ply=5,
        beats=[
            Beat("start", None, "Play on any site. Sidecoach reads the board from your screen."),
            Beat("said", "user_ask", "Ask it anything about the game, in your own words."),
            Beat("said", "stuck · L1", "Stuck? First a hint about the idea, never the answer."),
            Beat("said", "stuck · L2", "Still stuck? It narrows it down to the piece."),
            Beat("ply", 9, "Found it. It says why the move works."),
            Beat("ply", 13, "Checkmate. Then the whole game reviewed, one click away."),
        ],
    ),
    "hero": Story(
        full_from=8,
        beats=[
            Beat("start", None, "Play on any site. Sidecoach reads the board from your screen."),
            Beat("said", "stuck · L1", "Stuck? First a hint about the idea, never the answer."),
            Beat("said", "stuck · L2", "Still stuck? It points you at the piece and the question."),
            Beat("ply", 9, "Found it. It says why the move works."),
            Beat("ply", 13, "Checkmate. Then the whole game reviewed, one click away."),
        ],
    ),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / ("segoeuib.ttf" if bold else "segoeui.ttf")), size)


def _site_frames(rec: Path) -> tuple[list[float], list[np.ndarray]]:
    """The recording's stored frames with the time each was stored."""
    stamps = []
    for line in (rec / "index.jsonl").read_text(encoding="utf-8").splitlines():
        tick = json.loads(line)
        if tick["stored"] is not None:
            stamps.append(tick["ts"])
    cap = cv2.VideoCapture(str(rec / "frames.mkv"))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return stamps[: len(frames)], frames


def _at(stamps: list[float], t: float) -> int:
    i = int(np.searchsorted(np.asarray(stamps), t, side="right")) - 1
    return max(0, i)


def _window(img: Image.Image, title: str) -> Image.Image:
    w, h = img.size
    win = Image.new("RGB", (w, h + BAR_H), WINDOW_BAR)
    win.paste(img, (0, BAR_H))
    d = ImageDraw.Draw(win)
    for i, c in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse((10 + i * 18, 8, 20 + i * 18, 18), fill=c)
    d.text((w // 2, BAR_H // 2), title, font=_font(13), fill=BAR_TEXT, anchor="mm")
    mask = Image.new("L", win.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h + BAR_H - 1), radius=9, fill=255)
    out = Image.new("RGBA", win.size)
    out.paste(win, (0, 0), mask)
    return out


def compose(rec: Path, capture: Path, out: Path, story: str, width: int, crop: str) -> None:
    # The first game's plies, when each landed on the page (film_site.py).
    landed = json.loads((rec / "plies.json").read_text(encoding="utf-8"))
    if landed and isinstance(landed[0], list):  # a film of several games: one list each
        landed = landed[0]
    stamps, site = _site_frames(rec)
    panel = json.loads((capture / "frames.json").read_text(encoding="utf-8"))
    events = json.loads((capture / "events.json").read_text(encoding="utf-8"))
    p_stamps = [t for t, _ in panel]
    # The site's time 0 on the panel's clock: each ply of the first game landed at a known time
    # in the recording and was seen by the panel at a known wall time.
    seen: dict[int, float] = {}
    for e in events:
        if e["kind"] == "ply" and e["ply"]:
            seen.setdefault(e["ply"], e["t"])
    w0 = float(np.median([seen[k] - landed[k - 1] for k in seen if 0 < k <= len(landed)]))
    games = [e["t"] for e in events if e["kind"] == "game" and e["text"].startswith("new game")]

    def said(text: str) -> float | None:
        hits = [e["t"] for e in events if e["kind"] == "coach" and text in e["text"]]
        return hits[0] if hits else None

    plot = STORIES[story]
    beats: list[tuple[float, str]] = []
    for beat in plot.beats:
        if beat.kind == "start":
            t = p_stamps[0]
        elif beat.kind == "ply":
            t = seen.get(int(beat.key))  # type: ignore[arg-type]
        elif beat.kind == "game":
            n = int(beat.key)  # type: ignore[arg-type]
            t = games[n - 1] if n <= len(games) else None
        else:
            t = said(str(beat.key))
        if t is not None:
            beats.append((t + beat.after, beat.line))
    # What plays at full speed: a little before and after each moment.
    keep: list[tuple[float, float]] = []
    opening_ends = seen.get(plot.full_from, 0.0) - 0.6
    for e in events:
        if e["t"] < opening_ends:
            continue
        if e["kind"] == "ply":
            keep.append((e["t"] - 0.5, e["t"] + 1.2))
        elif e["kind"] == "coach":
            # Long enough to read it: a short word briefly, an answer by its length.
            words = len(str((e.get("data") or {}).get("full") or e["text"]).split())
            keep.append((e["t"] - 0.3, e["t"] + min(8.0, 2.2 + words / 6.0)))
    # From the moment the panel has a game on its board (an empty page is no first frame).
    start = games[0] + 0.5 if games else p_stamps[0] + 0.3
    if plot.start_ply and plot.start_ply in seen:
        start = seen[plot.start_ply] - 1.0
    # A few seconds past the last moment, before the panel is stopped.
    end = min(p_stamps[-1], max(b for _, b in keep) + 1.0) if keep else p_stamps[-1]
    plies = [e["t"] for e in events if e["kind"] == "ply" and e["ply"]]
    if plies:
        end = min(end, plies[-1] + 7.0)
    x0, y0, x1, y1 = (int(v) for v in crop.split(","))
    first_panel = Image.open(capture / "panel" / panel[0][1])
    pw, ph = first_panel.size
    site_h = ph
    site_w = round((x1 - x0) * site_h / (y1 - y0))
    canvas_w = GAP * 3 + site_w + pw
    floated = (capture / "captions.json").exists()
    # With the captions box: the line saying what happens goes on a band at the top, and the box
    # floats over the windows' lower edge (the real one keeps off the board); else the line
    # sits below the windows.
    top = 64 if floated else GAP
    canvas_h = top + BAR_H + ph + 74
    line_y = top // 2 if floated else canvas_h - 37
    font = _font(25, bold=True)
    small = _font(15, bold=True)
    frames: list[Image.Image] = []
    durations: list[int] = []
    t = start
    cache: dict[str, Image.Image] = {}
    # The app's captions box (record_live.py --captions), filmed at twice the pixels: floated
    # over the two windows, enlarged so its words read at the README's width.
    c_list = (
        json.loads((capture / "captions.json").read_text(encoding="utf-8"))
        if (capture / "captions.json").exists()
        else []
    )
    c_stamps = [ts for ts, _ in c_list]
    c_cache: dict[str, Image.Image | None] = {}
    c_width = round(canvas_w * 0.62)

    def caption_box(name: str) -> Image.Image | None:
        if name not in c_cache:
            img = Image.open(capture / "captions" / name).convert("RGB")
            arr = np.asarray(img).astype(int)
            # The box: whatever is lighter than the page behind it.
            ys, xs = np.where(arr.sum(axis=2) > 3 * 26)
            if len(xs) < 50:
                c_cache[name] = None
            else:
                box = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
                k = c_width / box.width
                c_cache[name] = box.resize((c_width, round(box.height * k)), Image.LANCZOS)
        return c_cache[name]

    if c_list:
        # Room below the windows for the tallest box the film shows, so it never covers a board.
        span = [n for ts, n in c_list if start - 1 <= ts <= end]
        tallest = max((b.height for n in span if (b := caption_box(n)) is not None), default=0)
        canvas_h = top + BAR_H + ph + tallest + 2 * GAP

    while t < end:
        fast = not any(a <= t <= b for a, b in keep)
        speed = (FAST * 2 if t < opening_ends else FAST) if fast else plot.pace
        sf = site[_at(stamps, t - w0)][y0:y1, x0:x1]
        simg = Image.fromarray(cv2.cvtColor(sf, cv2.COLOR_BGR2RGB)).resize(
            (site_w, site_h), Image.LANCZOS
        )
        name = panel[_at(p_stamps, t)][1]
        if name not in cache:
            cache[name] = Image.open(capture / "panel" / name).convert("RGB")
        canvas = Image.new("RGB", (canvas_w, canvas_h), BACKDROP)
        left = _window(simg, "The site you play on")
        right = _window(cache[name], "Sidecoach")
        canvas.paste(left, (GAP, top), left)
        canvas.paste(right, (GAP * 2 + site_w, top), right)
        if c_list:
            box = caption_box(c_list[_at(c_stamps, t)][1])
            if box is not None:
                bx = (canvas_w - box.width) // 2
                by = top + BAR_H + ph + GAP
                shadow = Image.new("RGBA", (box.width + 24, box.height + 24), (0, 0, 0, 0))
                ImageDraw.Draw(shadow).rounded_rectangle(
                    (12, 16, box.width + 12, box.height + 16), radius=22, fill=(0, 0, 0, 150)
                )
                canvas.paste(shadow, (bx - 12, by - 12), shadow)
                mask = Image.new("L", box.size, 0)
                ImageDraw.Draw(mask).rounded_rectangle(
                    (0, 0, box.width - 1, box.height - 1), radius=20, fill=255
                )
                canvas.paste(box, (bx, by), mask)
        d = ImageDraw.Draw(canvas)
        line = ""
        for bt, text in beats:
            if bt <= t + 0.2:
                line = text
        d.text((canvas_w // 2, line_y), line, font=font, fill=CAPTION, anchor="mm")
        if fast:
            # A fast-forward mark: two triangles and the speed.
            cx, cy = canvas_w - GAP - 22, line_y
            for dx in (-14, 0):
                d.polygon([(cx + dx, cy - 8), (cx + dx + 12, cy), (cx + dx, cy + 8)], fill=ACCENT)
            d.text((cx - 22, cy), f"{speed:g}×", font=small, fill=ACCENT, anchor="rm")
        scale = width / canvas_w
        frame = canvas.resize((width, round(canvas_h * scale)), Image.LANCZOS)
        frames.append(frame)
        durations.append(round(1000 / FPS))
        t += speed / FPS
    # The end holds.
    durations[-1] = 4000
    write_gif(frames, durations, out)


def write_gif(frames: list[Image.Image], durations: list[int], out: Path) -> None:
    """One palette for every frame (no flicker), frames that did not change merged."""
    picks = [frames[len(frames) * k // 8] for k in range(1, 8)]
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(picks)))
    for i, p in enumerate(picks):
        strip.paste(p, (0, i * frames[0].height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    # Merge frames that did not change.
    merged: list[Image.Image] = []
    merged_d: list[int] = []
    last = None
    for q, ms in zip(quantized, durations, strict=True):
        arr = np.asarray(q)
        if last is not None and np.array_equal(arr, last):
            merged_d[-1] += ms
            continue
        merged.append(q)
        merged_d.append(ms)
        last = arr
    merged[0].save(
        out,
        save_all=True,
        append_images=merged[1:],
        duration=merged_d,
        loop=0,
        optimize=True,
        disposal=1,
    )
    total = sum(merged_d) / 1000
    print(
        f"{out}: {len(merged)} frames, {total:.1f} s, {out.stat().st_size / 1e6:.1f} MB, "
        f"{frames[0].width}x{frames[0].height}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rec", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--story", default="hero", choices=sorted(STORIES))
    parser.add_argument("--width", type=int, default=1100)
    # The part of the site's frame shown: x0,y0,x1,y1 (the board and its players).
    parser.add_argument("--crop", default="196,60,756,720")
    args = parser.parse_args()
    compose(args.rec, args.capture, args.out, args.story, args.width, args.crop)


if __name__ == "__main__":
    main()
