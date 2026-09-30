"""Round 2 of our own piece set: one sheet from gpt-image-2.5, given three sets as examples
(Neo, staunty, tatiana) and a detailed brief; every piece inside its own square box. On the
ledger: run "pieces-r2", 10 cents cap (Tai 2026-09-30)."""

import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

REPO = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(REPO / "tools" / "eval"))
from judge import Ledger  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "r2"
OUT.mkdir(exist_ok=True)
for line in (REPO / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

ORDER = ["K", "Q", "R", "B", "N", "P"]
SHEET_ORDER = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"]
CELL = 128
LIGHT, DARK = (235, 236, 208), (115, 149, 82)


def piece(set_name: str, code: str) -> Image.Image:
    if set_name == "neo":
        return Image.open(HERE / "neo" / f"{code[0]}{code[1].lower()}.png").convert("RGBA")
    sheet = Image.open(REPO / "tests" / "fixtures" / "pieces" / f"{set_name}.png").convert("RGBA")
    k = SHEET_ORDER.index(code)
    return sheet.crop((k * 192, 0, (k + 1) * 192, 192))


def reference() -> Path:
    """The three example sets, one per row, each piece on its own board square, the square's
    edge drawn so the box each piece fills is plain."""
    sets = ["neo", "staunty", "tatiana"]
    img = Image.new("RGB", (CELL * 12, CELL * len(sets)), (38, 36, 33))
    for r, s in enumerate(sets):
        for c, code in enumerate([f"w{p}" for p in ORDER] + [f"b{p}" for p in ORDER]):
            sq = Image.new("RGBA", (CELL, CELL), LIGHT if (r + c) % 2 == 0 else DARK)
            sq.alpha_composite(piece(s, code).resize((CELL, CELL), Image.LANCZOS))
            ImageDraw.Draw(sq).rectangle((0, 0, CELL - 1, CELL - 1), outline=(60, 60, 60), width=1)
            img.paste(sq.convert("RGB"), (c * CELL, r * CELL))
    path = OUT / "reference.png"
    img.save(path)
    return path


PROMPT = """TASK
Design one complete, NEW 2D chess piece set for a chess app's board: 12 pieces (white and black king, queen, rook, bishop, knight, pawn). Deliver it as one sprite sheet laid out exactly as described under LAYOUT.

THE REFERENCE IMAGE
The attached image shows three existing piece sets, one set per row. In each row, left to right: white king, queen, rook, bishop, knight, pawn, then black king, queen, rook, bishop, knight, pawn. Each piece sits on its own square board cell, and the cell's edge is drawn. Study how every piece FILLS its square cell. Use the references for the qualities they share (chunky Staunton shapes, soft shading, thin dark outline, icon sizing). Your set must be clearly its own design: do NOT copy any row's silhouettes, knight head, crown, cross or mitre. It should look like a cleaner, simpler, more polished sibling of all three.

LAYOUT (follow exactly)
- Canvas 1536 x 1024 pixels, fully transparent background.
- The canvas is divided into a grid of 6 columns and 4 rows of SQUARE boxes, each box exactly 256 x 256 pixels, touching edge to edge, starting at the top-left corner.
- Draw each box's edge as a thin 2-pixel solid mid-grey (#808080) square outline. Draw all 24 boxes, including the empty ones.
- Row 1, boxes left to right: white king, white queen, white rook, white bishop, white knight, white pawn.
- Row 2, boxes left to right: black king, black queen, black rook, black bishop, black knight, black pawn.
- Rows 3 and 4: empty boxes (outline only, transparent inside).
- Exactly one piece per box. No piece, outline, highlight or shading may touch, cross or leave its box's edge.

SIZE AND PLACEMENT INSIDE EACH SQUARE BOX (the most important rule)
- Every piece is an icon scaled to fill its own square box, exactly like the references; do NOT draw the pieces at real-world relative sizes.
- Every piece's base sits on the same line: its bottom edge 22 px above the bottom of its box.
- Every piece is centred left to right in its box.
- Tops, measured from the top of the box: king 16 px, queen 18 px, bishop 20 px, rook 30 px, knight 22 px, pawn 44 px. So the pawn is only a little shorter than the others, never small.
- Widest point (the base) of every piece: 150 to 165 px, the same for all pieces within 15 px. The pieces must look equally heavy.

SHARED CONSTRUCTION (identical on all 12 pieces)
- Base: a wide, low, rounded plinth, about 160 px wide and 36 px tall, with one soft step above it. The same base on every piece.
- Collar: one simple rounded ring where the body meets the head, the same thickness on every piece.
- Outline: one continuous thin dark outline around each piece, 5 px, colour #1f1d1b, even weight, rounded joins. Inner detail lines (collar, crown band) are thinner, 3 px, the same colour.
- Shapes: smooth, simplified, chunky and slightly rounded; no fine carving, no engraved texture, no tiny ornaments. Readable at 40 px.

LIGHT AND COLOUR
- One light for the whole set, from the upper left. Each piece: lit on its left, a soft shadow band down its right side, a small soft highlight on the upper left of rounded parts. Gentle gradients, matte satin finish, no glossy hotspots, no reflections.
- White pieces: body #F3EFE6 (warm off-white); shadow side #CFC8BA; highlight #FFFFFF.
- Black pieces: body #4A4744 (warm charcoal, never pure black); shadow side #2F2D2B; a thin lighter rim along the lit left edge #6E6A66, so the black pieces read clearly on a dark square.

EACH PIECE
- King: tallest. Tapered body widening to a flared crown band, a smooth domed top, and a simple solid cross on top with short, thick, slightly rounded arms (cross about 40 px tall). No jewels.
- Queen: tapered body, a crown of five rounded points, each point ending in a small ball; a slightly larger ball on the centre top. The crown clearly different from the king's cross at a glance.
- Rook: a sturdy straight tower, slightly wider at the top, with three square battlements (two gaps) cut into the top; a thin band below the battlements.
- Bishop: a rounded mitre, like an egg standing on the collar, with one clean diagonal slit from upper right to lower left, and a small round ball on top.
- Knight: faces LEFT. A solid, friendly horse head and neck: rounded muzzle, one clear dark eye, one small upright ear, a smooth mane drawn as 3 to 4 simple broad scallops down the back of the neck. The head and neck sit on the same base as the other pieces. Calm expression, no teeth, no open mouth, no nostril detail beyond one small mark.
- Pawn: a round ball head on a short neck with the collar, on the shared base. Full and sturdy: its head about 70 px across.

DO NOT
- No board, no chequered squares, no background colour, no scenery, no ground shadow, no glow or halo, no reflections.
- No text, letters, numbers, labels or watermark anywhere.
- No 3D perspective or tilt: every piece seen straight from the front, flat 2D illustration.
- Do not draw any piece at real-world scale, and never draw a small pawn."""

PRICES = (8.0, 2.0, 30.0)  # $ per million: input (at the image rate, the higher), cached, image out
ledger = Ledger("pieces-r2", "piece-set sheet from examples", "gpt-image-2.5-sunburst", budget_usd=0.10, prices=PRICES, resume=True)

if __name__ == "__main__":
    ref = reference()
    if "--dry" in sys.argv:
        print(ref, len(PROMPT.split()), "words")
        sys.exit(0)
    name = sys.argv[1] if len(sys.argv) > 1 else "examples-1"
    ledger.check()
    t0 = time.time()
    with ref.open("rb") as f:
        r = httpx.post(
            "https://api.openai.com/v1/images/edits",
            headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            data={"model": "gpt-image-2.5-sunburst", "prompt": PROMPT, "size": "1536x1024", "quality": "high", "background": "transparent", "output_format": "png", "n": "1"},
            files={"image[]": ("reference.png", f, "image/png")},
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
