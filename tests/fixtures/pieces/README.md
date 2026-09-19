# Piece sets for the renderer

One sprite sheet per set: the twelve pieces in the order wK wQ wR wB wN wP bK bQ bR bB bN
bP, each 192 by 192 pixels with alpha, rasterised once from lichess's SVGs
(`lichess-org/lila`, `public/piece/<set>/`) by Chrome at that size. `tests/pieceset.py`
scales a sprite to the cell with area interpolation, which is what the site's own
rasterisation of the vector at the cell's size looks like to within the anti-aliasing.
The cburnett set is not here: python-chess carries its vectors, and `tests/pieceset.py`
rasterises them itself, in any colouring.

These are test fixtures only; nothing ships with them. Not here: `letter`, whose pawns
are dots the size of chess.com's legal-move hints, which the recogniser refuses to call
pieces by design; `monarchy`, missing from lila; `shahi-ivory-brown`, whose licence
forbids derivatives; `mono`, `disguised`, `horsey` and `xkcd`, novelties. Authors and
licences, from lila's COPYING.md:

| set | author | licence |
| --- | --- | --- |
| alpha | Eric Bentzen | free for personal non-commercial use |
| anarcandy | caderek | CC BY-NC-SA 4.0 |
| caliente | avi | CC BY-NC-SA 4.0 |
| california | Jerry S. | CC BY-NC-SA 4.0 |
| cardinal | sadsnake1 | CC BY-NC-SA 4.0 |
| celtic | Maurizio Monge | MIT |
| chess7 | Style-7 | freeware |
| chessnut | Alexis Luengas | Apache 2.0 |
| companion | David L. Brown | freeware |
| cooke | fejfar | CC BY-NC-SA 4.0 |
| dubrovny | sadsnake1 | CC BY-NC-SA 4.0 |
| fantasy | Maurizio Monge | MIT |
| firi | James Faure | CC BY 4.0 |
| fresca | sadsnake1 | CC BY-NC-SA 4.0 |
| gioco | sadsnake1 | CC BY-NC-SA 4.0 |
| governor | (lila) | see lila |
| icpieces | sadsnake1 | CC BY-NC-SA 4.0 |
| kiwen-suwi | neverRare | CC BY 4.0 |
| kosal | Kosal Sen | see lila |
| leipzig | Armando Hernandez Marroquin | freeware |
| maestro | sadsnake1 | CC BY-NC-SA 4.0 |
| merida | Armando Hernandez Marroquin | GPLv2+ |
| minimal-warmth | blunder_reign | CC BY-NC-SA 4.0 |
| mpchess | Maxime Chupin | GPLv3+ |
| papercut | Nikolay Anzarov | CC BY 4.0 |
| pirouetti | pirouetti | AGPLv3+ |
| pixel | therealqtpi | AGPLv3+ |
| reillycraig | Reilly Craig | see lila |
| rhosgfx | RhosGFX | CC0 1.0 |
| riohacha | (lila) | see lila |
| shapes | flugsio | CC BY-SA 4.0 |
| spatial | Maurizio Monge | MIT |
| staunty | sadsnake1 | CC BY-NC-SA 4.0 |
| tatiana | sadsnake1 | CC BY-NC-SA 4.0 |
| totoy | Kosal Sen | CC BY 4.0 |
