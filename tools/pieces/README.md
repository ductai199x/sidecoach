# Sidecoach's piece set

Our own chess pieces, replacing cburnett (2026-09-30). Tai wanted the clean, chunky, softly
shaded look of chess.com's Neo without a copy of it and without a share-alike licence: our
own drawing of the Staunton pattern (public domain since 1849).

## How it was made

1. **Concept sheets** from OpenAI's `gpt-image-2.5-sunburst` (1536x1024, high quality,
   transparent background; about 4-6 cents a sheet; every call on the `judge.Ledger`, runs
   `pieces-r1` .. `pieces-r6`, 46.1 cents in all).
   - Round 1, words only: rejected, pieces at real-world heights (a pawn half the king).
   - Round 2 (`gen_round2.py`): a reference image of three sets (Neo, staunty, tatiana),
     each piece on its board square, and a detailed brief: a 6x4 grid of 256 px square
     boxes, each piece's top and base in its box, the shared base, collar and outline, hex
     colours, one light from the upper left, a paragraph a piece, what not to draw.
     `round2-sheet.png`. Tai: "much closer". (Round 3, "less 3D", came out fat: rejected.)
   - Round 4 (`gen_round4.py`): round 2's own sheet as the input, "keep every shape, change
     only the rendering": flatter, a softer outline, a 4-peak queen (still with a centre ball).
   - Round 5 (`gen_round5.py`): round 4's sheet, "a tiny bit flatter", the queen's crown
     spelt out (four ball peaks, a V dip, nothing in the middle). `round5-sheet.png`, the set
     of 0.1.39-0.1.43.
   - Round 6 (`gen_round6.py`, 2026-10-01, 5.9 cents): round 5's sheet with no dark line
     inside a silhouette (Tai: "a lot of horizontal lines going across that are mostly just
     distractors ... the Neo set is cleaner, a bit more silhouette-like") -- the collars,
     bases and battlements told apart by tone -- and a little less flat (he found round 5 a
     bit too flat). `round6-sheet.png`: THE SET.
   The scripts record the prompts; they ran from a scratch folder, and the reference images
   of other sets (chess.com's among them) are not in the repository.
2. **Tracing** (`to_svg.py`): each piece found by its own shape (the model does not keep the
   grid it is asked for), split into colour layers -- the silhouette in the outline colour,
   then the body's tones from the lightest, each darker one over the lighter (round 6's soft
   shading has four tones on white, five on black) -- each smoothed and traced with potrace,
   stacked; all 12 at one scale, bases on one line, in a 512 viewBox. The shading is smoothed
   before it is cut into tones (SHADE_SIGMA; cut as it came, the black pieces streaked) and
   the tones' edges are blurred into each other inside the piece (an SVG blur clipped to the
   body, BLUR; without it the steps read as wood grain), the outline crisp. Checked by
   rendering each SVG and subtracting it from its source: 99.4-99.8% silhouette overlap,
   colour error 3.8-6 in 255.

## The other sets, the board colours, a player's own pieces

Settings > Board (2026-09-30) offers, besides ours, every Lichess set whose licence allows any
use (`fetch_sets.py` puts them in `desktop/static/pieces/<set>/`; authors and licences in
`desktop/static/pieces/LICENSE.txt`), 16 plain-colour board themes, and "Your own pieces":
image files the player chooses, kept only in their data folder (`src/sidecoach/looks.py`).
Nothing names, links to or fetches any other product's set. Every theme keeps the panel's
row-parity signature for the vision (tests/test_looks.py draws each and checks it).

## Lessons (Tai's)

- Board pieces are icons: each fills its own SQUARE box; never real-world relative heights.
- Give the model example images and a very detailed brief, not a paragraph of adjectives.
- When he likes the shapes, change only what he names.
- Clean like an icon: the outline only around the outside; inside, parts by tone, not lines.
