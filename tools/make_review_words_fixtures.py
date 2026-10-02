"""The AI coach's words on a reviewed game's key moves, as the desktop writes and links them, for
the phone's Game Review (`coach/review_words.py`, `coach/review_links.py`).

No model is called: a scripted one stands in. It reads the key move it is asked about out of the
request and answers from a script of canned replies made from that move's own facts -- good
paragraphs, a "why" that gives the better move away, a side named "you", a move given to the
wrong side, a piece the board puts elsewhere, a lesson naming a square, an answer without its
"why", odd heads and fences, empty and unreadable answers, failed calls -- each with token counts,
so the cost is exercised. Recorded for each game: the key moves, the estimate at several prices
and with moves already answered, and runs of `write` -- every request as the model was asked it
(system, messages, max_tokens, temperature, thinking), the replies it was handed, each
`on_answer`, and the result (texts, dropped, answers, model, cost, calls) or the error
raised -- with `linked` of the result's texts; and hand-written paragraphs on every key move that
hit every linking rule. Then the pieces on their own: `to_student`, `parse`, `in_colours` and
`checked` on the short game of the desktop's tests.

The games: the seven of profile/reviews.json (`make_review_fixtures.py`), each copied in whole so
this fixture stands on its own, the desktop tests' short game, and that game without a key move.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_review_words_fixtures.py
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.coach import review_links, review_words  # noqa: E402
from sidecoach.events import ReviewedGame  # noqa: E402
from sidecoach.llm.base import Completion  # noqa: E402
from sidecoach.llm.usage import Prices  # noqa: E402
from tests.test_review_words import _game as short_game  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
REVIEWS = HERE / "app" / "src" / "androidTest" / "assets" / "profile" / "reviews.json"
OUT = HERE / "app" / "src" / "androidTest" / "assets" / "coach" / "review_words.json"

USERS: list[str | None] = ["white", "black", None]
LUNA = Prices(0.20, 0.02, 1.20)
PRICES = [LUNA, Prices(1.0, None, 2.0), Prices()]
# An opening outside ASCII: the prompt is written with ensure_ascii=False.
ACCENTED = "Grünfeld Defence – Exchange Variation"


def prices_json(p: Prices) -> list[float | None]:
    return [p.prompt, p.cached, p.completion]


def opening_of(game: ReviewedGame) -> str | None:
    """`review_room._opening`: the opening the game reached, as the book last named it."""
    return next((m.after.opening for m in reversed(game.moves) if m.after.opening), None)


# --- the scripted model -------------------------------------------------------------------------


def stats_for(ply: int, kind: int) -> dict[str, Any]:
    """Token counts of several shapes: plain, cached, a cache write, floats, none at all."""
    shapes: list[dict[str, Any]] = [
        {"prompt_tokens": 1200 + 13 * ply, "completion_tokens": 400 + 7 * ply},
        {
            "prompt_tokens": 1500 + ply,
            "cached_tokens": 512,
            "completion_tokens": 700 + ply,
            "reasoning_tokens": 300,
            "finish_reason": "stop",
        },
        {"prompt_tokens": 1100 + 2 * ply, "cache_write_tokens": 256, "completion_tokens": 650},
        {"prompt_tokens": 1000.0 + ply, "completion_tokens": 321.0},
        {},
    ]
    return shapes[kind % len(shapes)]


def facts_of(content: str) -> dict[str, Any]:
    return json.loads(content)["key move"]


def other_square(game: ReviewedGame, ply: int) -> str:
    """The square of the other side's king before the move: the mover's king is never there."""
    board = chess.Board(game.moves[ply - 1].fen_before)
    king = board.king(not board.turn)
    return chess.square_name(king if king is not None else chess.E4)


def reply_for(variant: str, game: ReviewedGame, ply: int, km: dict[str, Any]) -> str:
    """A canned answer of one kind, made from the key move's facts as the model was told them."""
    mover = km["played by"].split(" ")[0]
    other = "Black" if mover == "White" else "White"
    san = km["move"]
    chances = km[f"{mover}'s winning chances"]
    better = km.get(f"{mover}'s better move")
    line = km.get("its line") or []
    reply = km.get(f"{other}'s best reply")
    bad = km["kind"] in review_words.BAD
    lesson = "Before you move, look at what your opponent can do next, and answer it first."
    if variant == "good":
        why = f"{mover} played {san}: {km['what it does']}. {mover}'s winning chances: {chances}."
        if bad and reply:
            why += f" {other} can answer with {reply}, and {mover} is worse after it."
        out = f"### why\n{why}\n"
        if bad and better:
            out += f"### better\n{better} was stronger: the line runs {' '.join(line)}.\n"
        return out + f"### lesson\n{lesson}"
    if variant == "pronouns":
        why = (
            f"{mover} also keeps their pieces together while {other} needs time; {other}'s pieces "
            f"and their squares matter. The {mover} king stays safe, and {mover} blocks the file "
            f"and leaves {other}'s pawns weak. If {other} takes, {mover} recaptures, and "
            f"{other} are not ready: {other} have to defend, which gives {mover} chances."
        )
        lesson = "When your opponent threatens something, deal with it first."
        return f"## Why\n{why}\n\n## Lesson\n{lesson}"
    if variant == "reveal":
        named = better or san
        why = (
            f"{mover} should have played {named} instead of {san}; {other} replies and "
            f"{mover} suffers."
        )
        return f"### why\n{why}\n### better\n{named} keeps the balance.\n### lesson\n{lesson}"
    if variant == "you":
        why = f"You played {san}, and your opponent now has the easier game."
        return f"### why\n{why}\n### lesson\n{lesson}"
    if variant == "wrong side":
        why = f"{other} played {san} too early, and the position suffered."
        return f"### why\n{why}\n### lesson\n{lesson}"
    if variant == "board":
        return (
            f"### why\n{mover}'s king on {other_square(game, ply)} is exposed after {san}.\n"
            f"### lesson\n{lesson}"
        )
    if variant == "lesson square":
        why = f"{mover} played {san}, and the game turned on it."
        lesson = f"Keep an eye on {other_square(game, ply)} and on White's plans."
        return f"### why\n{why}\n### lesson\n{lesson}"
    if variant == "no why":
        return f"### better\n{better or san} was better.\n### lesson\n{lesson}"
    if variant == "empty":
        return ""
    if variant == "unread":
        return "The move was interesting, and it changed the game a lot."
    if variant == "fenced":
        return (
            "```\n## WHY:\n"
            f"{mover}\u2019s move {san} changes the position: {km['what it does']}.\n"
            "```\n#### Better \u2014\n"
            f"{better or san} first, as {mover} could.\n"
            '### "lesson"\nKeep your pieces protected.\n### extra\nignored words'
        )
    raise ValueError(variant)


VARIANTS = (
    "good",
    "pronouns",
    "reveal",
    "you",
    "good",
    "wrong side",
    "board",
    "fail",
    "lesson square",
    "no why",
    "fenced",
    "empty",
    "good",
    "unread",
)


class Scripted:
    """A model that answers by a script: the variant for each ply, or an error. Records every
    request as it was asked. No `model`, no `with_effort`: the phone's fake has neither."""

    name = "scripted"

    def __init__(self, game: ReviewedGame, script: dict[int, str]) -> None:
        self.game = game
        self.script = script
        self.lock = threading.Lock()
        self.requests: dict[int, dict[str, Any]] = {}
        self.replies: dict[int, dict[str, Any]] = {}

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(
        self,
        system: str,
        messages: Any,
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        assert system == review_words.SYSTEM
        km = facts_of(messages[0]["content"])
        ply = km["ply"]
        with self.lock:
            self.requests[ply] = {
                "messages": [dict(m) for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "thinking": thinking,
            }
        variant = self.script[ply]
        if variant.startswith("error:"):
            message = variant[len("error:") :]
            with self.lock:
                self.replies[ply] = {"error": message}
            raise RuntimeError(message)
        text = reply_for(variant, self.game, ply, km)
        stats = stats_for(ply, ply)
        with self.lock:
            self.replies[ply] = {"text": text, "stats": stats}
        return Completion(text=text, stats=dict(stats))


class Reasoning(Scripted):
    """The same, as a model that reasons by its effort (OpenAI's): `model`, and `with_effort`
    recording the effort it was set to. The phone's is an OpenAICompatProvider on a scripted
    wire."""

    model = "scripted-luna"

    def __init__(self, game: ReviewedGame, script: dict[int, str]) -> None:
        super().__init__(game, script)
        self.efforts: list[str | None] = []

    def with_effort(self, effort: str | None) -> Reasoning:
        self.efforts.append(effort)
        return self


def words_json(words: review_words.ReviewWords) -> dict[str, Any]:
    return {
        "texts": {str(p): parts for p, parts in words.texts.items()},
        "dropped": {str(p): why for p, why in words.dropped.items()},
        "answers": {str(p): text for p, text in words.answers.items()},
        "model": words.model,
        "cost": words.cost,
        "calls": words.calls,
    }


def run(
    name: str,
    game: ReviewedGame,
    *,
    user: str | None,
    opening: str | None,
    prices: Prices,
    script: dict[int, str],
    thinking: bool = False,
    answered: dict[int, str] | None = None,
    keep: bool = True,
    keep_fails: int | None = None,
    model: type[Scripted] = Scripted,
) -> dict[str, Any]:
    """One `write` with the scripted model, everything it was asked and gave recorded."""
    fake = model(game, script)
    kept: dict[str, Any] = {}

    def on_answer(ply: int, text: str, stats: dict[str, Any]) -> None:
        kept[str(ply)] = {"text": text, "stats": stats}
        if ply == keep_fails:
            raise OSError("the store is locked")  # not kept only costs a resume

    out: dict[str, Any] = {
        "name": name,
        "user": user,
        "opening": opening,
        "prices": prices_json(prices),
        "thinking": thinking,
        "answered": None if answered is None else {str(p): t for p, t in answered.items()},
        "keep": keep,
        "keep_fails": keep_fails,
    }
    try:
        words = review_words.write(
            game,
            fake,
            user=user,
            opening=opening,
            prices=prices,
            thinking=thinking,
            answered=answered,
            on_answer=on_answer if keep else None,
        )
        out["result"] = words_json(words)
        texts = {str(p): parts for p, parts in words.texts.items()}
        out["links"] = review_links.linked(texts, game, user)
    except Exception as exc:  # noqa: BLE001 -- every call failed: the first failure raised
        out["raised"] = str(exc)
    out["replies"] = {str(p): r for p, r in sorted(fake.replies.items())}
    out["requests"] = {str(p): r for p, r in sorted(fake.requests.items())}
    out["on_answer"] = {p: kept[p] for p in sorted(kept, key=int)}  # by ply, not by arrival
    if isinstance(fake, Reasoning):
        out["efforts"] = fake.efforts
    return out


def script_for(plies: list[int], offset: int) -> dict[int, str]:
    out = {}
    for i, ply in enumerate(plies):
        variant = VARIANTS[(i + offset) % len(VARIANTS)]
        out[ply] = f"error:429: slow down on ply {ply}" if variant == "fail" else variant
    return out


# --- hand-written paragraphs for the links -----------------------------------------------------

TERMS_SAID = (
    "A fork, two forks, it forked and forking; a pin, pins, pinned, pinning; a skewer, skewered. "
    "A discovered attack, discovered checks, a double attack, the back rank and the back-rank; a "
    "sacrifice, sacrificed, sacrificing; passers and passed pawns; an isolated pawn; doubled "
    "pawns; a backward pawn; hanging pawns; a hanging piece and a loose one; a trapped bishop; "
    "outposts; the open file, open c-files; the half-open file and the semi-open e-file; the "
    "bishop pair, the pair of bishops; a bad bishop; the initiative; a tempo, two tempi; "
    "zugzwang; prophylaxis, prophylactic; a fianchetto, fianchettoed; king safety; a pawn break; "
    "the pawn majority, a queenside majority; promotion, promotes, queening; deflection, "
    "deflected; a decoy; overloaded, overloading; interference; an in-between move, a "
    "zwischenzug, an intermezzo; quiet moves; counterplay; a closed centre, the closed center; "
    "the opposition; weak squares and holes; the exchange. Capitals: Fork, PIN, Half-Open D-file."
)
STRUCTURES_SAID = (
    "Your doubled pawns and your opponent's isolated pawn; White's passed pawns and Black's "
    "backward c-pawn; their hanging pawns; the passed e-pawn and the doubled f-pawns; my "
    "isolated d-pawn; your doubled c-pawns; Black's passed a-pawn."
)
SQUARES_SAID = (
    "The diagonal a1-h8, the c4-square, e4, E4 and b-file, B-file and the d-file; the king on "
    "e1, the Black queen on d8, your opponent's knight at f3, White's light-squared bishop on "
    "c4, their extra pawn on a2, the passed pawn on e5, the Pawn On E4, the b4 square, h9 and i4."
)
MOVES_SAID = "Qh5+, Nxf7#, O-O, O-O-O, e8=Q, exd5, Kxe2, Ra1, Bb5 and Nc3 came up; Q, N and B."


def link_paragraphs(game: ReviewedGame, ply: int) -> dict[str, str]:
    """A key move's paragraphs to link: its own move, the better move's line, the reply's line,
    a threat of the side not to move, its pieces and pawn pushes, and the words above."""
    move = game.moves[ply - 1]
    before = chess.Board(move.fen_before)
    said = [f"The move {move.san} was played."]
    line = before.copy(stack=False)
    for uci in move.assessment.best[:6]:
        played = chess.Move.from_uci(uci)
        if played not in line.legal_moves:
            break
        said.append(f"{line.san(played)} (or {line.san(played).rstrip('+#')})")
        line.push(played)
    after = chess.Board(move.after.fen)
    for uci in (move.after.lines[0].moves[:6] if move.after.lines else [])[:6]:
        played = chess.Move.from_uci(uci)
        if played not in after.legal_moves:
            break
        said.append(after.san(played))
        after.push(played)
    passed = chess.Board(move.after.fen)
    if not passed.is_check():
        passed.push(chess.Move.null())
        threat = next(iter(passed.legal_moves), None)
        if threat is not None:
            said.append(f"a threat {passed.san(threat)}")
    pushes = [
        chess.square_name(m.to_square)
        for m in before.legal_moves
        if before.piece_type_at(m.from_square) == chess.PAWN and not before.is_capture(m)
    ]
    if pushes:
        said.append(f"the move {pushes[0]} and the square {pushes[-1]}")
    for square, piece in sorted(before.piece_map().items())[:4]:
        owner = "White's" if piece.color == chess.WHITE else "Black's"
        said.append(f"{owner} {chess.piece_name(piece.piece_type)} on {chess.square_name(square)}")
    why = "; ".join(said) + ". " + SQUARES_SAID + " " + MOVES_SAID
    return {
        "why": why,
        "better": TERMS_SAID,
        "lesson": "A fork and a pin need loose pieces; your passed pawns win endgames on e4.",
        "extra": STRUCTURES_SAID,
    }


# --- the pieces on their own --------------------------------------------------------------------

STUDENT = [
    "White's bishop on c4 eyes f7, so Black must defend.",
    "Black has a strong knight, and Black also threatens Nd4.",
    "Black is a pawn up; White was careless.",
    "This gives Black chances.",
    "The Black king is safe; the white king is not, and White pieces hang.",
    "If Black takes, White recaptures and White doesn't lose time.",
    "White's winning chances fall because Black's knight reaches e4.",
    "Black can reorganize their pieces.",
    "White must meet Black's threat before pursuing their own plans.",
    "Black's rooks and their files are open.",
    "White forks the queen and the pawn. Black can save only one of them.",
    "Black's king and rook are not as active as they could be.",
    "White are down a queen, and White need to act.",
    "Black blocks the file and leaves White's queen safe.",
    "Black neutralizes the threat and stabilizes the position.",
    "Black not only escapes the check but also solidifies the defense.",
    "This allows Black to regroup and opens lines.",
    "Black first improves the rook, waits for the reply, and then uses it.",
    "White makes a threat Black cannot meet, then the knight reaches g6 and attacks.",
    "The plan for Black is to push.",
    "The position is forcing, meaning Black has no quiet moves.",
    "White forces Black to defend.",
    "A clear win for Black, who is up a queen.",
    "Black has a strong knight.",
    "White\u2019s pawn on e4 is weak, and Black\u2019s knight hits it.",
    "Black doesn't need to hurry, and Black isn't in danger; White hasn't castled.",
    "White, who was attacking, has to retreat.",
    "Black has the initiative; White are in trouble and White have to defend; "
    "White don't have time.",
    "Because Black also still controls d5, White's knight lacks squares.",
    "Black quickly develops and castles, and Black's rooks connect.",
    "Black fixes the pawns and watches the queen, or Black tries something else.",
    "White misses the tactic and loses material; White then resigns.",
    "When White plays Nf3, Black answers Nc6 and Black equalizes.",
    "The White position is solid, the black structure weak, and the White forces scattered.",
    "Between White and Black, the knights trade; among them the f-pawns matter, and one of "
    "their rooks is idle.",
    "White's queen and their rook are strong. They dominate.",
    "Black: the side to move. White? The side waiting! Black never gives up.",
    "white's pawn and black's bishop; the WHITE king.",
    "If White lets Black castle, Black stays safe and White loses the thread \u2014 White "
    "regroups.",
    "Black, who has the bishop pair, plays for a win; White, who was worse, defends and holds.",
    "Black's pieces and White's pawns; they move, and their plans change.",
    "Then White pushes, and Black reacts but Black falls behind.",
    "Black often plays e5 here and Black sometimes plays c5 instead.",
    "Now White carries the attack and wins the exchange; White also finishes it.",
    "Black is better. White is worse. Black wins.",
    "",
    "No side at all is named here.",
]

PARSED = [
    "### why\nA\nand B\n### Better:\nC\n```\n### lesson\nD\n```",
    "no heads here",
    "### better\nC",
    "### WHY\n\n  a  b \n### lesson:\nc",
    "#### Better \u2014 x\ny\n### why\nz",
    "# why\n",
    "###why\nx",
    "### whyever\nx",
    '  ## "why"\nx\n\t### lesson\ny',
    "### why\r\nA\r\n### lesson\r\nB\r\n",
    "### why\nA\n### why\nB",
    "text before\n### why\nA",
    "### why\n```\nA\n```",
    "",
    "\n",
    "### why",
    "###### why\nA\n####### lesson\nB",
    "### why the move mattered\nA\n### lesson (in short)\nB",
    "### why\nA \u00a0B\u2003C",
    "x ### why\nA",
]

COLOURED: list[tuple[Any, str]] = [
    ({"piece": "your bishop on c4", "key": "If you let them, their best is Nxe5"}, "White"),
    ("Bf5 and Bd6 among them", "Black"),
    ("between you and them, one of their pawns", "White"),
    ("YOUR king and Yours; THEIRS and They", "Black"),
    (["you", "them", "your queen", "of them"], "White"),
    ({"a": "your", "b": ["they", "you"], "c": True, "d": None, "e": 3}, "Black"),
    ("Among you and Of their rooks", "White"),
    ("youth and theirselves; you're, they've", "Black"),
]


def checks() -> list[dict[str, Any]]:
    """`checked` on the desktop tests' short game, the tests' cases and more."""
    game = short_game()
    cases: list[tuple[dict[str, str], int, str | None]] = [
        ({"why": "Your knight on c6 guards e5.", "lesson": "Guard your centre."}, 4, "black"),
        ({"why": "Black's isolated f-pawn on f2 is weak."}, 4, "black"),
        ({"why": "Bc4 was the move.", "better": "Bc4 develops first and eyes f7."}, 3, "white"),
        ({"why": "Bc4 was the move.", "better": "Bc4 develops first and eyes f7."}, 3, "black"),
        ({"why": "Bc4 was the move.", "better": "Bc4 develops first and eyes f7."}, 3, None),
        (
            {
                "why": "Black's knight on c6 guards Black's pawn on e5.",
                "better": "x",
                "lesson": "Before you chase a piece, see what your opponent threatens.",
            },
            4,
            "black",
        ),
        (
            {
                "why": "Black's knight on c6 guards Black's pawn on e5.",
                "lesson": "Black should guard e5 first.",
            },
            4,
            "black",
        ),
        ({"why": "Black played Nf3 too early."}, 3, "white"),
        ({"why": "Black could have played Bc4."}, 3, "white"),
        ({"why": "White's Bc4 develops first.", "better": "Black must stop Bc4."}, 3, "white"),
        (
            {"why": "White played Nf3.", "better": "Your Bc4 was better.", "lesson": "On e4."},
            3,
            "black",
        ),
        (
            {
                "why": "Black blundered with Nd4: White's Nxd4 wins a piece.",
                "better": "Nf6 is safe.",
            },
            6,
            "black",
        ),
        ({"why": "Black blundered with Nd4, and Nf6 was the move.", "better": "Nf6."}, 6, "black"),
        ({"why": "Black blundered with Nd4, and Nf6 was the move."}, 6, "white"),
        ({"better": "Nf6 was the move."}, 6, "black"),
        ({}, 6, "black"),
    ]
    out = []
    for parts, ply, user in cases:
        kept, dropped = review_words.checked(parts, game, ply, user)
        out.append(
            {
                "parts": parts,
                "ply": ply,
                "user": user,
                "kept": kept,
                "dropped": dropped,
                "wrong_side": {
                    part: review_words.wrong_side(text, game, ply) for part, text in parts.items()
                },
            }
        )
    return out


# --- the games ----------------------------------------------------------------------------------


def quiet(game: ReviewedGame) -> ReviewedGame:
    """The game with no key move: every label taken off."""
    copy = game.model_copy(deep=True)
    for move in copy.moves:
        move.assessment.label = None
    return copy


def still(value: Any) -> Any:
    """The JSON with every event's clock (`ts`) at 0: a game built here is the same each run."""
    if isinstance(value, dict):
        return {k: 0.0 if k == "ts" else still(v) for k, v in value.items()}
    if isinstance(value, list):
        return [still(v) for v in value]
    return value


def record(name: str, game: ReviewedGame, side: str | None, real: bool) -> dict[str, Any]:
    plies = review_words.key_plies(game)
    opening = opening_of(game)
    out: dict[str, Any] = {
        "name": name,
        "user_side": side,
        "reviewed": game.model_dump(mode="json") if real else still(game.model_dump(mode="json")),
        "opening": opening,
        "key_plies": plies,
    }
    out["prompts"] = {
        str(user): {str(p): review_words.prompt(game, p, user, opening) for p in plies}
        for user in USERS
    }
    estimates = []
    for user in USERS:
        for prices in PRICES:
            for answered in (set(), set(plies[:1]), set(plies)):
                for said in (opening, ACCENTED) if user == side else (opening,):
                    estimates.append(
                        {
                            "user": user,
                            "opening": said,
                            "prices": prices_json(prices),
                            "answered": sorted(answered),
                            "cost": review_words.estimate(game, user, said, prices, answered),
                        }
                    )
    out["estimates"] = estimates

    runs = []
    for i, user in enumerate(USERS):
        runs.append(
            run(
                f"mixed for {user}",
                game,
                user=user,
                opening=opening if user != side else ACCENTED,
                prices=LUNA,
                script=script_for(plies, 5 * i + len(name)),
            )
        )
    if plies:
        first = plies[:2]
        said = "### why\nWhite played the move at ply {}, and Black's reply holds."
        answered = {p: said.format(p) for p in first}
        answered[999] = "### why\nNot a key move of this game."
        runs.append(
            run(
                "resumed",
                game,
                user=side,
                opening=opening,
                prices=Prices(1.0, None, 2.0),
                script=script_for(plies, 3),
                answered=answered,
                keep_fails=plies[-1],
            )
        )
    if not real:
        every = dict.fromkeys(plies, "error:500: the server is down")
        runs.append(
            run("every call fails", game, user=side, opening=None, prices=LUNA, script=every)
        )
        runs.append(
            run(
                "every call fails, some answered",
                game,
                user=side,
                opening=None,
                prices=LUNA,
                script=every,
                answered={plies[0]: "### why\nWhite played Nf3 early."} if plies else {},
            )
        )
        runs.append(
            run(
                "thinking, prices unknown, nothing kept",
                game,
                user=side,
                opening=opening,
                prices=Prices(),
                script=script_for(plies, 0),
                thinking=True,
                keep=False,
            )
        )
        runs.append(
            run(
                "every move answered before",
                game,
                user=side,
                opening=None,
                prices=LUNA,
                script={},
                answered={p: "### why\nBlack's knight left e5 alone." for p in plies},
            )
        )
    out["runs"] = runs

    links = []
    for user in USERS:
        texts = {str(p): link_paragraphs(game, p) for p in plies}
        linked = review_links.linked(texts, game, user)
        links.append({"user": user, "texts": texts, "linked": linked})
    out["links"] = links
    return out


def effort(game: ReviewedGame, side: str | None) -> dict[str, Any]:
    """A model that reasons by its effort, told "none": its model's name kept with the words."""
    plies = review_words.key_plies(game)
    script = {p: ("good", "reveal", "you", "pronouns")[i % 4] for i, p in enumerate(plies)}
    return run(
        "effort none",
        game,
        user=side,
        opening=opening_of(game),
        prices=LUNA,
        script=script,
        model=Reasoning,
    )


def main() -> None:
    fixture = json.loads(REVIEWS.read_text(encoding="utf-8"))
    games = []
    for g in fixture["games"]:
        game = ReviewedGame.model_validate(g["reviewed"])
        games.append(record(g["name"], game, g["user_side"], real=True))
        print(f"{g['name']}: {len(games[-1]['key_plies'])} key moves", flush=True)
    short = short_game()
    games.append(record("the tests' short game", short, "black", real=False))
    games.append(record("the short game without a key move", quiet(short), "black", real=False))
    real = ReviewedGame.model_validate(fixture["games"][1]["reviewed"])

    out = {
        "system": review_words.SYSTEM,
        "answer_tokens": review_words.ANSWER_TOKENS,
        "games": games,
        "effort": {"game": 1, **effort(real, fixture["games"][1]["user_side"])},
        "to_student": [
            {"user": user, "paragraph": p, "read": review_words.to_student(p, user)}
            for p in STUDENT
            for user in USERS
        ],
        "parse": [{"text": t, "parts": review_words.parse(t)} for t in PARSED],
        "in_colours": [
            {"value": v, "mover": m, "out": review_words.in_colours(v, m)} for v, m in COLOURED
        ],
        "checks": checks(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    runs = sum(len(g["runs"]) for g in games) + 1
    requests = sum(len(r["requests"]) for g in games for r in g["runs"])
    print(
        f"{OUT}: {len(games)} games, {runs} runs, {requests} requests, "
        f"{OUT.stat().st_size // 1024} KB"
    )


if __name__ == "__main__":
    main()
