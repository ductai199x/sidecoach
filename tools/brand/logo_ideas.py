"""Logo concepts for Sidecoach through OpenAI's Images API (gpt-image-2.5), for choosing
among, never for shipping: an app icon must be crisp at 16 px, so the chosen concept is
rebuilt by hand as an SVG.

    uv run python tools/brand/logo_ideas.py --dry-run             # the estimate, no call
    uv run python tools/brand/logo_ideas.py                       # first pass: 8 concepts x 2
    uv run python tools/brand/logo_ideas.py --only knight-lens knight-wave --quality high --n 4

Every call goes on the ledger (data/eval/usage/, the judge's format, purpose "logo") and the
run stops at --budget-usd. Images land under data/brand/<stamp>/ with their prompt beside
them, and a contact sheet (sheet.png, on the app's dark grey) shows them all at once.
Pricing (2026-09-21, developers.openai.com/api/docs/pricing): output $30 per million
tokens, text input $5; a 1024x1024 image is about 300 tokens at low, 1,100 at medium,
4,200 at high (gpt-image-1's counts, which the response's `usage` corrects per call).
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.config import load_dotenv  # noqa: E402

ENDPOINT = "https://api.openai.com/v1/images/generations"
USAGE_MINE = ROOT / "data" / "eval" / "usage" / f"usage-{os.getpid()}.jsonl"
PRICE_OUT_PER_M = 30.0
PRICE_IN_PER_M = 5.0
TOKENS_PER_IMAGE = {"low": 300, "medium": 1100, "high": 4200, "xhigh": 8000, "max": 12000}

# The brief every concept shares: the docs' own words for a transparent logo, plus what an
# icon needs (simple, two colours, no text) and the palette the app will use.
STYLE = (
    " Flat vector-style app icon mark: a single bold symbol, at most two colours -- warm"
    " amber (#F2B441) for the main shape and deep navy (#141A2A) for details -- that would"
    " also work as a one-colour silhouette. Simple geometry, thick even strokes, no gradients,"
    " no shading, no 3D, no text, no letters, no numbers. Fully transparent background."
    " Deliver a single centered logo with generous padding, clean alpha edges, and no solid"
    " backdrop, scenery, checkerboard, or watermark. Hard crisp edges: no glow, no blur, no"
    " soft shadow, no outer halo. It must read clearly at 16 pixels and at 1024."
)

CONCEPTS: dict[str, str] = {
    "knight-bubble": (
        "Logo for a chess coaching app that talks to the player: the silhouette of a chess"
        " knight in profile, and cut out of it in negative space, a speech bubble where the"
        " mouth would be, so the knight is speaking."
    ),
    "knight-lens": (
        "Logo for a chess coaching app that watches the screen: a chess knight in profile"
        " whose single eye is a camera lens -- a ring with a small circle inside -- bold and"
        " geometric."
    ),
    "knight-wave": (
        "Logo for a chess coaching app that listens and speaks: the silhouette of a chess"
        " knight built from horizontal bars of different lengths, like a sound wave or an"
        " audio equalizer, stacked to form the knight's shape."
    ),
    "side-squares": (
        "Logo for an app called Sidecoach, a chess coach at your side: two rounded squares"
        " side by side and touching; the left one deep navy with a small amber pawn, the"
        " right one amber with a small navy speech bubble."
    ),
    "king-shadow": (
        "Logo for a chess coaching app, the coach standing beside the player: a chess king"
        " piece, and behind it, offset to one side and slightly larger, a second king drawn"
        " as an outline only, like a mentor standing at its shoulder."
    ),
    "question-pawn": (
        "Logo for a chess coaching app that asks before it tells: a chess pawn whose round"
        " head and neck curve into the hook of a question mark, with the pawn's base as the"
        " dot -- one continuous bold shape."
    ),
    "eye-board": (
        "Logo for a chess coaching app that reads the board from the screen: an almond-"
        "shaped eye whose iris is a small chessboard of four by four squares in amber and"
        " navy."
    ),
    "knight-s": (
        "Logo for an app called Sidecoach: a chess knight in profile whose mane and neck"
        " form one bold letter-S curve, so the knight's outline and the S are the same"
        " stroke; abstract and minimal."
    ),
    # The second pass (Tai's pick, 2026-09-21): the knight with the lens eye, four ways.
    "lens-head": (
        "App icon for a chess coaching app that watches the screen: only the head and neck"
        " of a chess knight in profile, facing left, cropped to fill a square composition"
        " with no base; its single eye is a camera lens drawn as one thick navy ring with a"
        " navy dot inside, large enough to be the focal point; one clean continuous outline."
    ),
    "lens-tile": (
        "App icon for a chess coaching app that watches the screen: a navy rounded square"
        " tile, and on it in amber the head and neck of a chess knight in profile facing"
        " left, its single eye a camera lens drawn as one thick ring with a dot inside; the"
        " knight fills the tile with even margins."
    ),
    "lens-clean": (
        "Logo for a chess coaching app that watches the screen: a full chess knight piece in"
        " profile facing left, one amber silhouette with a simple two-step base, its single"
        " eye a camera lens drawn as one thick navy ring with a navy dot inside; a single"
        " navy stroke for the mane; nothing behind it."
    ),
    "lens-beside": (
        "Logo for a chess coaching app, a coach at your side that watches the screen: the"
        " head and neck of an amber chess knight in profile facing left with a camera-lens"
        " eye (one thick navy ring with a dot), and behind it, offset up and to the right, a"
        " second knight head in solid navy, like a mentor at its shoulder."
    ),
}


class Ledger:
    """Every paid call on the record; the run stops at the budget."""

    def __init__(self, run: str, model: str, budget_usd: float) -> None:
        self.run, self.model, self.budget = run, model, budget_usd
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.spent = 0.0
        self.inflight = 0.0  # the estimate of calls not yet answered
        self._lock = threading.Lock()
        USAGE_MINE.parent.mkdir(parents=True, exist_ok=True)

    def begin(self, estimate: float) -> bool:
        with self._lock:
            if self.spent + self.inflight + estimate > self.budget:
                return False
            self.inflight += estimate
            return True

    def record(self, item: str, usage: dict, estimate: float, elapsed: float) -> float:
        tin = int(usage.get("input_tokens") or 0)
        tout = int(usage.get("output_tokens") or 0)
        cost = tin / 1e6 * PRICE_IN_PER_M + tout / 1e6 * PRICE_OUT_PER_M
        with self._lock:
            self.calls += 1
            self.tokens_in += tin
            self.tokens_out += tout
            self.spent += cost
            self.inflight -= estimate
            with USAGE_MINE.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "run": self.run,
                            "purpose": "logo",
                            "model": self.model,
                            "item": item,
                            "prompt_tokens": tin,
                            "cached_tokens": 0,
                            "completion_tokens": tout,
                            "cost_usd": round(cost, 4),
                            "elapsed_s": round(elapsed, 2),
                        }
                    )
                    + "\n"
                )
        return cost

    def summary(self) -> str:
        return (
            f"{self.calls} calls, {self.tokens_in} in / {self.tokens_out} out tokens,"
            f" ${self.spent:.2f} of the ${self.budget:.2f} budget"
        )


def generate(
    client: httpx.Client, key: str, model: str, prompt: str, n: int, quality: str, size: str
) -> tuple[list[bytes], dict]:
    body = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "size": size,
        "quality": quality,
        "background": "transparent",
        "output_format": "png",
        "moderation": "low",
    }
    response = client.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {key}"},
        json=body,
    )
    if response.status_code != 200:
        raise RuntimeError(f"{response.status_code}: {response.text[:400]}")
    data = response.json()
    images = [base64.b64decode(d["b64_json"]) for d in data.get("data", [])]
    return images, data.get("usage") or {}


def contact_sheet(folder: Path, labels: list[tuple[str, Path]], cell: int = 256) -> Path:
    """Every image on the app's dark grey with its name under it, one PNG."""
    from PIL import Image, ImageDraw

    cols = 4
    rows = (len(labels) + cols - 1) // cols
    pad, text_h = 16, 22
    sheet = Image.new(
        "RGB", (cols * (cell + pad) + pad, rows * (cell + pad + text_h) + pad), "#262421"
    )
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(labels):
        image = Image.open(path).convert("RGBA")
        image.thumbnail((cell, cell))
        x = pad + (i % cols) * (cell + pad)
        y = pad + (i // cols) * (cell + pad + text_h)
        sheet.paste(image, (x + (cell - image.width) // 2, y + (cell - image.height) // 2), image)
        draw.text((x, y + cell + 4), label, fill="#bababa")
    out = folder / "sheet.png"
    sheet.save(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--model", default="gpt-image-2.5-sunburst")
    parser.add_argument("--quality", default="medium", choices=list(TOKENS_PER_IMAGE))
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--n", type=int, default=2, help="renders per concept")
    parser.add_argument("--only", nargs="*", default=None, help="concept names to run")
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default=None, help="folder (default data/brand/<stamp>)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("OPENAI_API_KEY", "")
    concepts = {k: v for k, v in CONCEPTS.items() if not args.only or k in args.only}
    unknown = set(args.only or []) - set(CONCEPTS)
    if unknown:
        parser.error(f"unknown concepts: {sorted(unknown)}; known: {sorted(CONCEPTS)}")
    per_image = TOKENS_PER_IMAGE[args.quality] / 1e6 * PRICE_OUT_PER_M
    per_call = per_image * args.n + 200 / 1e6 * PRICE_IN_PER_M
    estimate = per_call * len(concepts)
    print(
        f"{len(concepts)} concepts x {args.n} at {args.quality} on {args.model}:"
        f" about ${estimate:.2f} (${per_image:.3f} an image), cap ${args.budget_usd:.2f}"
    )
    if args.dry_run:
        return 0
    if not key:
        print("no OPENAI_API_KEY in the environment or .env", file=sys.stderr)
        return 2
    if estimate > args.budget_usd:
        print("the estimate is over the cap; raise --budget-usd or narrow --only", file=sys.stderr)
        return 2

    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = Path(args.out) if args.out else ROOT / "data" / "brand" / stamp
    folder.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(stamp, args.model, args.budget_usd)
    made: list[tuple[str, Path]] = []
    lock = threading.Lock()

    def one(name: str) -> str:
        prompt = concepts[name] + STYLE
        if not ledger.begin(per_call):
            return f"{name}: skipped, the cap would be passed"
        begun = time.perf_counter()
        with httpx.Client(timeout=300.0) as client:
            try:
                images, usage = generate(
                    client, key, args.model, prompt, args.n, args.quality, args.size
                )
            except Exception as exc:  # noqa: BLE001 - one failed concept is reported, not fatal
                ledger.inflight -= per_call
                return f"{name}: FAILED {exc}"
        elapsed = time.perf_counter() - begun
        cost = ledger.record(name, usage, per_call, elapsed)
        paths = []
        for k, image in enumerate(images, 1):
            path = folder / f"{name}-{k}.png"
            path.write_bytes(image)
            paths.append(path)
        (folder / f"{name}.json").write_text(
            json.dumps(
                {
                    "prompt": prompt,
                    "model": args.model,
                    "quality": args.quality,
                    "size": args.size,
                    "usage": usage,
                    "cost_usd": round(cost, 4),
                    "elapsed_s": round(elapsed, 1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        with lock:
            made.extend((f"{name} {k}", p) for k, p in enumerate(paths, 1))
        return (
            f"{name}: {len(images)} images, {usage.get('output_tokens')} out tokens,"
            f" ${cost:.3f}, {elapsed:.0f}s"
        )

    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for line in pool.map(one, list(concepts)):
            print(line, flush=True)
    if made:
        made.sort(key=lambda item: item[0])
        print("sheet:", contact_sheet(folder, made))
    print(ledger.summary())
    print("folder:", folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
