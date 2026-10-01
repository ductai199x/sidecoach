"""Round 6: round 5's sheet with no lines inside the silhouettes (the parts told apart by tone,
not by drawn lines) and a little less flat. Tai 2026-10-01 00:00: "the pieces are a bit too
flat ... less lines inside of their silhouette ... a lot of horizontal lines going across that
are mostly just distractors ... the Neo set is cleaner, a bit more silhouette-like", "1 more
sheet". Ledger run "pieces-r6", 8 cents cap, ONE call."""

import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(REPO / "tools" / "eval"))
from judge import Ledger  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "r6"
OUT.mkdir(exist_ok=True)
for line in (REPO / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

PROMPT = """TASK
Redraw the attached chess piece sprite sheet. It is our own set; the art director approves every shape. Keep every shape exactly as it is, and make only the two changes below.

THE ART DIRECTOR'S WORDS
"The pieces are a bit too flat, and I want them to have fewer lines inside their silhouette. Right now they have a lot of horizontal lines going across that are mostly just distractors. It should be cleaner, a bit more silhouette-like."

KEEP EXACTLY THE SAME
- Every silhouette, shape, proportion, size and position of all 12 pieces: the same heights and widths, the same slim bodies, bases, collars, necks and heads. Do not make any piece wider, fatter, squatter, rounder, bigger or smaller. Trace the existing outer shapes.
- The queen's crown: exactly four peaks, each ending in one small ball, a low V-shaped dip between the two inner peaks, nothing in the centre.
- The layout: white pieces in the top row (king, queen, rook, bishop, knight, pawn), black pieces in the second row, the lower rows empty. Transparent background.
- The outer outline: the same dark warm grey (#3A3633) line all the way round the OUTSIDE of each piece, the same thickness as now.
- The colours: white pieces #F6F3EE with shadow #D9D3C8; black pieces #57534F with shadow #3F3C39.

CHANGE 1: NO LINES INSIDE THE SILHOUETTE (all 12 pieces)
- The only dark line on a piece is its outer outline.
- Remove every dark line that crosses the inside of a piece: the lines above and below each collar ring, the line along the top of each base, the line under the rook's battlements, the line where a head meets its collar, and every other horizontal line inside the shape.
- Where two parts of a piece meet (head and collar, collar and body, body and base), show it only by tone: a soft shadow of the piece's own shadow colour just under the part above, or the lower part a shade darker. Never a drawn line.
- Keep only these small inner details, as they are now: the knight's eye and nostril, and the bishop's diagonal slit.
- Each piece reads as one clean, solid, filled shape, like a modern flat icon with gentle shading.

CHANGE 2: A LITTLE LESS FLAT (all 12 pieces)
- A little more volume than now, about halfway back towards softly shaded: on each part, the main colour blends into the shadow colour over a short, smooth gradient toward the right side, instead of a hard flat band.
- Round parts (the pawn's head, the bishop's mitre and its top ball, the queen's crown balls, the king's head) get one small, soft highlight on the upper left.
- Still clean and simple: no gloss, no shine streaks, no reflections, no rim light, no bevel, no texture.

DO NOT
No board, no squares, no box outlines, no text, no background, no ground shadow, no glow or halo, no perspective. Do not redesign, widen, fatten, shorten or move any piece. Do not add any new line or detail."""

PRICES = (8.0, 2.0, 30.0)
ledger = Ledger(
    "pieces-r6",
    "round-5 sheet, no inner lines, a little less flat",
    "gpt-image-2.5-sunburst",
    budget_usd=0.08,
    prices=PRICES,
    resume=True,
)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "clean"
    ledger.check()
    t0 = time.time()
    with (HERE / "r5" / "flatter.png").open("rb") as f:
        r = httpx.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            data={
                "model": "gpt-image-2.5-sunburst",
                "prompt": PROMPT,
                "size": "1536x1024",
                "quality": "high",
                "background": "transparent",
                "output_format": "png",
                "n": "1",
            },
            files={"image[]": ("round5.png", f, "image/png")},
            timeout=600,
        )
    elapsed = time.time() - t0
    body = r.json()
    if r.status_code != 200:
        print(json.dumps({"error": body.get("error", body), "status": r.status_code}))
        sys.exit(1)
    usage = body.get("usage", {})
    usd = ledger.record(
        name,
        {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        },
        elapsed,
    )
    (OUT / f"{name}.png").write_bytes(base64.b64decode(body["data"][0]["b64_json"]))
    print(json.dumps({"name": name, "usd": round(usd, 4), "usage": usage, "s": round(elapsed)}))
    print(f"spent this run: ${ledger.cost():.4f}")
