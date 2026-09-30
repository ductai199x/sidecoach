"""Round 4: round 2's sheet redrawn with the SAME shapes, only flatter and cleaner, and a
queen's crown of 4 peaks (Tai 2026-09-30 15:39). Ledger run "pieces-r4", 8 cents cap."""

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
OUT = HERE / "r4"
OUT.mkdir(exist_ok=True)
for line in (REPO / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

PROMPT = """TASK
Redraw the attached chess piece sprite sheet. It is our own set and the art director likes its shapes. Keep EVERYTHING about the shapes exactly as they are and change only the rendering, plus one detail of the queen.

THE ART DIRECTOR'S WORDS
"I just want the same pieces but a bit more flat and clean looking, and the queen's crown should have 4 peaks instead of 5 peaks."

KEEP EXACTLY THE SAME (do not change these at all)
- Every silhouette, outline shape and proportion of all 12 pieces: the same heights, the same widths, the same slim, elegant bodies, the same bases, collars, necks and heads. Do NOT make any piece wider, fatter, squatter, rounder or bigger. Trace the existing shapes.
- The same position of every piece on the canvas, the same sizes, the same layout: white pieces in the top row (king, queen, rook, bishop, knight, pawn), black pieces in the second row, the lower rows empty.
- The king's cross, the rook's battlements, the bishop's mitre with its slit and ball, the knight's head, ear, eye and mane shape, the pawn.
- Transparent background.

CHANGE 1: FLATTER AND CLEANER RENDERING (all 12 pieces)
- Flat 2D vector style, like a clean modern app icon set, not a 3D render.
- Each piece: one flat main colour, plus ONE soft shadow tone as a smooth band along the right side of each part, plus at most one small flat highlight on the upper left of the head. Nothing else.
- Remove all glossy gradients, shine, specular highlights, rim light, reflections, bevels and embossed edges.
- Outline: keep one even outline around each piece, a little thinner and softer than now: 4 px, dark warm grey #3A3633 instead of black. Keep inner lines only where parts meet (collar rings, base step), 3 px, same colour.
- White pieces: main #F6F3EE, shadow #D9D3C8. Black pieces: main #57534F, shadow #3F3C39, a subtle lighter edge #6C6864 only where needed for the shape to read. Never pure black.

CHANGE 2: THE QUEEN'S CROWN (both queens)
- The crown has exactly 4 peaks instead of 5, each peak ending in a small round ball, evenly spaced and symmetric (two outer peaks and two inner ones), plus the ball on the centre top as before if it fits; keep the crown the same width and height as now.

DO NOT
No board, no squares, no box outlines, no text, no background, no ground shadow, no glow or halo, no perspective. Do not redesign any piece."""

PRICES = (8.0, 2.0, 30.0)
ledger = Ledger("pieces-r4", "round-2 shapes, flatter, 4-peak queen", "gpt-image-2.5-sunburst", budget_usd=0.08, prices=PRICES, resume=True)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "flat-r2"
    ledger.check()
    t0 = time.time()
    with (HERE / "r2" / "examples-1.png").open("rb") as f:
        r = httpx.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            data={"model": "gpt-image-2.5-sunburst", "prompt": PROMPT, "size": "1536x1024", "quality": "high", "background": "transparent", "output_format": "png", "n": "1"},
            files={"image[]": ("round2.png", f, "image/png")},
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
