"""Round 5: round 4's sheet, a tiny bit flatter, and the queen's crown fixed (4 peaks, nothing
in the middle). Tai 2026-09-30 15:42. Ledger run "pieces-r5", 8 cents cap."""

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
OUT = HERE / "r5"
OUT.mkdir(exist_ok=True)
for line in (REPO / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

PROMPT = """TASK
Redraw the attached chess piece sprite sheet. It is our own set; the art director approves it. Keep everything exactly as it is, and make only the two changes below.

THE ART DIRECTOR'S WORDS
"It's better. The queen is still messed up. If you can, a tiny bit flatter would be perfect."

KEEP EXACTLY THE SAME
- Every silhouette, shape, proportion, size and position of all 12 pieces: the same heights and widths, the same slim bodies, bases, collars, necks and heads. Do not make any piece wider, fatter, squatter, rounder, bigger or smaller. Trace the existing shapes.
- The layout: white pieces in the top row (king, queen, rook, bishop, knight, pawn), black pieces in the second row, the lower rows empty. Transparent background.
- The outline: the same thin, soft, dark warm grey outline (#3A3633), the same inner lines.
- The colours: white pieces #F6F3EE with shadow #D9D3C8; black pieces #57534F with shadow #3F3C39.

CHANGE 1: A TINY BIT FLATTER (all 12 pieces)
- Each part of each piece has exactly two flat tones: the main colour, and one shadow tone as a band along its right side with a crisp, clean edge (only a slight softening, no long gradient).
- Remove any remaining gradient, glossy highlight, shine, reflection, rim light, bevel or 3D rounding effect. At most one small flat highlight shape on the upper left of a round head.
- The result should look like clean flat vector icons.

CHANGE 2: THE QUEEN'S CROWN (white queen and black queen)
- The crown has EXACTLY FOUR peaks and nothing else on top.
- The four peaks are evenly spaced along the top of the crown and symmetric about the centre line: two outer peaks leaning slightly outward, and two inner peaks standing straight up, all four the same height and shape.
- Each of the four peaks ends in one small round ball.
- In the centre of the crown, between the two inner peaks, there is only a low V-shaped dip: NO fifth peak, NO centre ball, NO stem, NO spike, NO jewel.
- Keep the crown's overall width and height and the rest of the queen (body, collar, base) exactly as they are now.

DO NOT
No board, no squares, no box outlines, no text, no background, no ground shadow, no glow or halo, no perspective. Do not redesign any piece."""

PRICES = (8.0, 2.0, 30.0)
ledger = Ledger("pieces-r5", "round-4 sheet, flatter, 4-peak queen fixed", "gpt-image-2.5-sunburst", budget_usd=0.08, prices=PRICES, resume=True)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "flatter"
    ledger.check()
    t0 = time.time()
    with (HERE / "r4" / "flat-r2.png").open("rb") as f:
        r = httpx.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            data={"model": "gpt-image-2.5-sunburst", "prompt": PROMPT, "size": "1536x1024", "quality": "high", "background": "transparent", "output_format": "png", "n": "1"},
            files={"image[]": ("round4.png", f, "image/png")},
            timeout=600,
        )
    elapsed = time.time() - t0
    body = r.json()
    if r.status_code != 200:
        print(json.dumps({"error": body.get("error", body), "status": r.status_code}))
        sys.exit(1)
    usage = body.get("usage", {})
    usd = ledger.record(name, {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0)}, elapsed)
    (OUT / f"{name}.png").write_bytes(base64.b64decode(body["data"][0]["b64_json"]))
    print(json.dumps({"name": name, "usd": round(usd, 4), "usage": usage, "s": round(elapsed)}))
    print(f"spent this run: ${ledger.cost():.4f}")
