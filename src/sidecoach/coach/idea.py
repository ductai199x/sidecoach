"""What the best line is for: the goal it reaches, the blow that reaches it, and its target.

Stockfish gives the best line and its score; the tagger reads what each move does. The coach
named the idea from the position's static features and the line's first move alone, so a
fork on the player's second move was never named -- a static pin elsewhere on the board took
the headline -- and the first word on a mate in two was "what is your opponent's most
dangerous idea?" (Tai's puzzles, 2026-09-23). This reads the line through the player's own
moves: the goal (a mate in N, the material won, a promotion, a threat to meet), the tactic
that delivers it on whichever move of the line it comes, how the line starts (a check, a
trade, a sacrifice), and the target (their king, or what falls).

No square here is named by the coach before the disclosure level allows it: `coach/facts.py`
turns this into words, each at its level.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import chess

from sidecoach.analysis import exchange, motifs
from sidecoach.coach import plans
from sidecoach.events import Analysis, MotifTag

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
_COUNTED = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)

# Won and level, by the engine's expected score. `Analysis.decisive` is not "won": it marks
# a critical moment, where only one move holds -- an equality puzzle is exactly that, and
# reading it as won said "you can win their queen" and "a winning attack" of positions the
# engine scored level (Lichess equality puzzles, 2026-09-23).
WON = 0.8
LEVEL = 0.62
# A line that nets this much (pawns' worth) within its first HORIZON plies wins material;
# a pawn is a win too when the position is won (a pawn up into a won ending is the point).
WIN_GAIN = 2
# Below this expected score, material the opponent hands over is theirs to give, not the
# player's win (`_first_gift`).
GIFT_SCORE = 0.3
HORIZON = 16
# A won position whose line checks this often in the player's first four moves, winning no
# material in sight, is an attack on the king: "no tactic here" was said of a rook
# sacrifice that opened one (Lichess 0199s, 2026-09-23).
ATTACK_CHECKS = 2
# The furthest of the player's moves a named pattern may land on and still be the idea.
FAR_BLOW = 3

# The named mating patterns, which say more than "a mating attack".
MATE_PATTERNS = (
    "smotheredMate",
    "backRankMate",
    "anastasiaMate",
    "arabianMate",
    "hookMate",
    "bodenMate",
    "doubleBishopMate",
    "dovetailMate",
    "vukovicMate",
    "killBoxMate",
    "swallowstailMate",
    "epauletteMate",
    "triangleMate",
    "blindSwineMate",
    "cornerMate",
    "operaMate",
    "pillsburysMate",
    "morphysMate",
    "balestraMate",
)
# The blow, the most telling first: what a coach names as the point of the line.
BLOW_RANK: tuple[str, ...] = (
    *MATE_PATTERNS,
    "doubleCheck",
    "fork",
    "skewer",
    "discoveredAttack",
    "absolutePin",
    "pin",
    "trappedPiece",
    "capturingDefender",
    "deflection",
    "attraction",
    "hangingPiece",
    "promotion",
    "mate",
)
_BLOW_INDEX = {name: i for i, name in enumerate(BLOW_RANK)}
# The tactics aimed at a piece: the blow only when the line then takes that piece. A
# discovered attack on a pawn the line never touches, a fork of two pieces that both get
# What a mate is named by: its pattern, or two checks at once. A pin or a skewer that
# happens to be on the board as the mate lands is not its point ("It's a pin" was said of
# Qh7+ Kf6 Qxh6#, Lichess 01xbu).
MATE_BLOWS = (*MATE_PATTERNS, "doubleCheck")
# What can clear a pawn's way or hold the defence off, for a promotion.
PROMOTION_BLOWS = (
    "discoveredAttack",
    "deflection",
    "capturingDefender",
    "absolutePin",
    "pin",
    "attraction",
    "skewer",
    "fork",
    "doubleCheck",
)
# The tactics that make a pawn won in a decided position the point of the line.
PRIMARY = (
    "doubleCheck",
    "fork",
    "skewer",
    "discoveredAttack",
    "absolutePin",
    "pin",
    "trappedPiece",
)
# What a move before the blow does to prepare it.
PREPARATION = ("sacrifice", "deflection", "attraction", "capturingDefender", "clearance")


@dataclass(frozen=True)
class Idea:
    """The point of the best line, from the side to move's view."""

    goal: str  # "mate", "win", "promote", "defend", "plan"
    first: chess.Move | None
    line: tuple[chess.Move, ...] = ()
    mate_in: int | None = None
    gain: int = 0
    won: str | None = None  # "their queen", "a rook", "the exchange", "a knight and a pawn"
    blow: MotifTag | None = None  # squares as on `blow_board`
    blow_move: chess.Move | None = None
    blow_board: chess.Board | None = None
    blow_at: int = 1  # which of the player's moves carries the blow: 1 is the first
    # "win": the ply of the player's move that first wins it (0 is the first move): O-O and
    # then Nxd4 wins the pawn with the second move, not the first (Tai's game, 2026-09-24).
    gain_ply: int = 0
    preparation: tuple[str, ...] = ()  # PREPARATION tags of the moves before the blow
    starts_with: str = "quiet"  # "check", "capture", "sacrifice", "promotion", "quiet"
    target: chess.Square | None = None  # their king (mate), or what falls, on the board now
    threat: chess.Move | None = None  # "defend": the opponent's move if the player passed
    threatened: str | None = None  # "mate", or the piece the threat wins: "your rook"
    # "defend": the threat is a quiet move that would threaten it next, `follow` the move
    # that would then mate or take (Rh1, then Qh8#: the master games, 2026-09-24).
    prepared: bool = False
    follow: chess.Move | None = None
    flight: tuple[chess.Square, ...] = ()  # "mate": their king's free squares now
    checks: int = 0  # how many checks the side to move has
    level: bool = False  # the line only holds the balance: what it wins, it wins back
    down: bool = False  # the player is behind in material now
    draw: str | None = None  # "draw": "perpetual", "stalemate" or "repetition"
    # A won endgame's technique: "convert" the ending traded into ("pawn", "rook", ...);
    # "king" whether the king move takes the opposition; "breakthrough" the pawn that runs.
    ending: str | None = None
    opposition: bool = False
    runner: chess.Square | None = None
    # Why it works, beyond the pattern: a piece of theirs that could take the striking
    # piece but is pinned (the "pin" of Lichess's fork puzzles), where it stands when the
    # blow lands; and whether the line holds an in-between move (a check before taking
    # back, Lichess "intermezzo").
    pinned_guard: chess.Square | None = None
    intermezzo: bool = False
    # A second tactic the line also cashes in, besides the blow: the fork that is also a
    # discovered attack (Lichess tags both; the coach named one).
    also: str | None = None
    pv_moves: tuple[str, ...] = field(default_factory=tuple)
    # The player's sacrifices on the way to the point: (ply, what is given), "queen",
    # "rook", "exchange", "bishop", "knight", "pawn" -- material the reply keeps.
    sacrifices: tuple[tuple[int, str], ...] = ()
    winning: bool = False  # the engine calls the position won
    score: float | None = None  # the best line's expected score, for the player
    # "win": what the line takes is mostly what the exchange in progress took from the player
    # (`exchange_owed`) -- won back, not won: a knight taken back after they took one is a
    # trade (Tai's puzzles, 2026-09-26: "you can win a knight" for 3.Nxh5 Rxh5).
    back: bool = False


def exchange_owed(board: chess.Board) -> dict[int, int]:
    """The exchange the last move belongs to, from the side to move's view: the run of
    captures that ends with the opponent's last move, counted per kind of piece -- how many
    more of the side to move's it took than of the opponent's. Returned only when that leaves
    the side to move owed material (they took last; it is the player's to take back): taking
    it back is winning back, not winning. Empty when the last move took nothing or the board
    does not know its moves (a position set up from a FEN)."""
    if not board.move_stack:
        return {}
    us = board.turn
    before = board.copy()
    net: dict[int, int] = {}
    while before.move_stack:
        move = before.pop()
        if not before.is_capture(move):
            break
        if before.is_en_passant(move):
            victim = chess.PAWN
        else:
            victim = before.piece_type_at(move.to_square)
            if victim is None:
                break
        # Popped, the board has the capturer to move: the opponent took one of the player's.
        net[victim] = net.get(victim, 0) + (1 if before.turn != us else -1)
    net = {kind: n for kind, n in net.items() if n}
    return net if worth(net) > 0 else {}


def worth(pieces: dict[int, int]) -> int:
    """What a count of pieces by kind is worth, in pawns."""
    return sum(VALUES.get(kind, 0) * n for kind, n in pieces.items())


def read(board: chess.Board, analysis: Analysis, *, owed: dict[int, int] | None = None) -> Idea:
    """The idea of `analysis`'s best line from `board` (the position it measured), with
    the sacrifices the player makes on the way to its point. `owed` is what the exchange in
    progress took from the player (`exchange_owed`, read from the board's own moves when not
    given): material the line takes up to that is won back, not won."""
    owed = exchange_owed(board) if owed is None else owed
    idea = _read(board, analysis, owed)
    if idea.first is None:
        return idea
    boards = [board.copy(stack=False)]
    for move in idea.line:
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    pv = list(idea.line)
    if idea.goal == "mate" and idea.mate_in:
        upto = 2 * idea.mate_in - 1
    elif idea.goal in ("win", "combination"):
        upto = 2 * idea.blow_at if idea.blow is not None else HORIZON
        gain, at = line_gain(boards, pv, min(len(pv), HORIZON))
        upto = max(upto, _first_reaching(boards, pv, at, min(gain, WIN_GAIN)) if gain else 0)
    elif idea.goal in ("attack", "promote"):
        upto = 8
    else:
        upto = 2
    top = analysis.lines[0]
    winning = (top.mate is not None and top.mate > 0) or top.expected_score >= WON
    return replace(
        idea,
        pinned_guard=_pinned_guard(idea, boards, pv),
        also=_also(idea, boards, pv),
        intermezzo=_intermezzo(boards, pv, min(len(pv), 10)),
        sacrifices=_sacrifices(boards, pv, upto),
        winning=winning,
        level=top.expected_score < LEVEL and not (top.mate is not None and top.mate > 0),
        down=_balance(board, board.turn) < 0,
        score=top.expected_score,
    )


def _read(board: chess.Board, analysis: Analysis, owed: dict[int, int]) -> Idea:
    pv = _legal_prefix(board, analysis.lines[0].moves) if analysis.lines else []
    us = board.turn
    checks = sum(1 for move in board.legal_moves if board.gives_check(move))
    if not pv:
        return Idea(goal="plan", first=None, checks=checks)
    boards = [board.copy(stack=False)]
    for move in pv:
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    first = pv[0]
    starts_with = _start(board, boards, pv)
    mate = analysis.lines[0].mate
    if mate is not None and mate > 0:
        end = min(len(pv), 2 * mate - 1)
        blow, blow_at, preparation = _blow(boards, pv, end, mate, MATE_BLOWS)
        king = board.king(not us)
        return Idea(
            goal="mate",
            first=first,
            line=tuple(pv),
            mate_in=mate,
            blow=blow[1] if blow else None,
            blow_move=pv[blow[0]] if blow else None,
            blow_board=boards[blow[0]] if blow else None,
            blow_at=blow_at,
            preparation=preparation,
            starts_with=starts_with,
            target=king,
            flight=_flight(board, not us),
            checks=checks,
        )
    second = analysis.lines[1].expected_score if len(analysis.lines) > 1 else None
    draw = _drawn(boards, pv, analysis.lines[0].expected_score, second)
    if draw is not None:
        return Idea(
            goal="draw",
            first=first,
            line=tuple(pv),
            starts_with=starts_with,
            target=board.king(not us),
            checks=checks,
            draw=draw,
        )
    horizon = min(len(pv), HORIZON)
    # The player queening: the material before it is what a win wins; after it, what the
    # new queen does, and the pawn is the point (a race won with Kxb8, Lichess 02YZr).
    queened = next(
        (j + 1 for j, move in enumerate(pv[:horizon]) if j % 2 == 0 and move.promotion), None
    )
    reach = queened - 1 if queened else horizon
    # In a position the engine calls lost, what they give up on the way is given for their
    # attack, not won: counted only up to their first piece left to be taken ("you can win
    # back a knight and a pawn" at 29.exf4 in Johner-Nimzowitsch, the knight Ng4+ given to
    # open the king).
    if analysis.lines[0].expected_score < GIFT_SCORE:
        reach = min(reach, _first_gift(boards, pv, reach))
    gain, at = line_gain(boards, pv, reach)
    peak = at
    # A pawn of the player's reaching the seventh, or queening, anywhere in the line.
    seventh = 6 if us == chess.WHITE else 1
    # A pawn of the player's queening in the line; one reaching the seventh only in an
    # ending, where it is on its way: in a middlegame a pawn on the seventh is a capture
    # on f7 in an attack ("you can push a pawn through to promotion" was said of exf7 in
    # Gligoric-Fischer, move 17).
    ending = _pieces_left(board) <= 4
    queens = any(
        move.promotion
        or (
            ending
            and boards[j].piece_type_at(move.from_square) == chess.PAWN
            and chess.square_rank(move.to_square) == seventh
        )
        for j, move in enumerate(pv[:horizon])
        if j % 2 == 0
    )
    # A pawn is a win when the game is better for it; two pawns' worth always.
    better_for_it = analysis.lines[0].expected_score >= LEVEL
    # What they just took counts against the line (`exchange_owed`): what it nets decides
    # whether it wins, by the same rule as any win; a line that nets less only wins back.
    net = gain - worth(owed)
    back = (
        bool(owed)
        and gain > 0
        and not (net >= WIN_GAIN or (net >= 1 and better_for_it and not queens))
    )
    # The blow is what wins it: looked for up to where the line first wins that much, not
    # where it has picked up the most (a fork five moves after the queen fell was named). A
    # net win is first won where the net is reached, past taking back what they took.
    wins_at = worth(owed) + min(net, WIN_GAIN) if owed and not back else min(gain, WIN_GAIN)
    first_at = _first_reaching(boards, pv, at, wins_at) if gain > 0 else 0
    if owed and not back and first_at > 2 * FAR_BLOW:
        # A net win first reached past the player's third move is the engine's line talking,
        # not the point: taking back is ("you can win a queen for a rook and a bishop" of a
        # queen given up on the fourteenth ply after 1...Nxe3 2.fxe3, Lichess 00LZf).
        back = True
        first_at = _first_reaching(boards, pv, at, min(gain, WIN_GAIN))
    at = first_at or at
    gained = _gained_kinds(board, _line_end(boards, pv, at) if at else boards[-1], us)
    blow, blow_at, preparation = _blow(boards, pv, at or min(len(pv), 6), None, gained=gained)
    if blow is not None and blow_at > FAR_BLOW:
        # A loose rook picked up on the fifth move is not the idea a player can see from
        # here; the start of the line is (Tai's game g06, 2026-09-24).
        blow, blow_at, preparation = None, 1, ()
    winning = analysis.lines[0].expected_score >= WON
    # A win the score denies: the engine calls the game level, the player is not behind,
    # and the material comes only down the line -- "you can win a knight and two pawns for a
    # bishop" was said at move 6 of the Ruy Lopez (Capablanca-Marshall, 2026-09-24).
    first_win = max(0, at - 1 if (at - 1) % 2 == 0 else at - 2) if gain > 0 else 0
    phantom = (
        not better_for_it
        and _balance(board, us) + worth(owed) >= 0
        and first_win > 0
        and not (analysis.lines[0].mate is not None and analysis.lines[0].mate > 0)
    )
    if not phantom and (gain >= WIN_GAIN or (gain >= 1 and better_for_it and not queens)):
        # A pawn is a win too when the position turns on it and a real tactic takes it: a
        # discovered check that nets a pawn into a won ending is still the point.
        # A line that stops on the player's capture: what takes back is counted as given,
        # as `line_gain` counts it ("you win the exchange", not "a rook", after Nxe1).
        won, target = _won(board, _line_end(boards, pv, at), us, pv[:at], boards)
        if back:
            # Won back: the most valuable piece the line takes within the player's first
            # FAR_BLOW moves ("their queen" where they took the queen and the line takes a
            # bishop and then theirs); never one far down the line (00LZf's queen at ply 14).
            near = min(peak, 2 * FAR_BLOW, len(pv))
            won = _top_taken(board, _line_end(boards, pv, near), us) or won
        elif owed:
            # A net win, named net: where they took a bishop and the line takes a knight and
            # two pawns, the win is two pawns (Tai, 2026-09-26: "it's not winning shit").
            netted, where = _won(board, _line_end(boards, pv, at), us, pv[:at], boards, owed)
            won, target = netted or won, where if netted else target
        # The player's move that first wins it: the last of theirs within `at` plies.
        gain_ply = max(0, at - 1 if (at - 1) % 2 == 0 else at - 2)
        return Idea(
            goal="win",
            first=first,
            line=tuple(pv),
            gain=gain,
            gain_ply=gain_ply,
            won=won,
            back=back,
            blow=blow[1] if blow else None,
            blow_move=pv[blow[0]] if blow else None,
            blow_board=boards[blow[0]] if blow else None,
            blow_at=blow_at,
            preparation=preparation,
            starts_with=starts_with,
            target=target,
            checks=checks,
        )
    if queens:
        # The pawn queening is the point, anywhere in the line, or the first move takes it
        # to the seventh: what holds the defence off is the blow, when there is one -- a
        # pin, a deflection; never a discovered attack on some pawn elsewhere.
        upto = queened if queened is not None else min(len(pv), 6)
        blow, blow_at, preparation = _blow(boards, pv, upto, None, PROMOTION_BLOWS)
        return Idea(
            goal="promote",
            first=first,
            line=tuple(pv),
            blow=blow[1] if blow else None,
            blow_move=pv[blow[0]] if blow else None,
            blow_board=boards[blow[0]] if blow else None,
            blow_at=blow_at,
            preparation=preparation,
            starts_with=starts_with,
            checks=checks,
        )
    checks_given = sum(1 for j in range(0, min(len(pv), 8), 2) if boards[j].gives_check(pv[j]))
    # Won, or clearly better with material given for the checks: a sacrificial attack at
    # 0.78 was "which of your pieces has no job right now?" (Lichess PZm0M).
    better = analysis.lines[0].expected_score >= 0.65
    their_king = board.king(not us)
    near_king = their_king is not None and any(
        chess.square_distance(pv[j].to_square, their_king) <= 2
        for j in range(0, min(len(pv), 8), 2)
    )
    opened = better and bool(_sacrifices(boards, pv, 6)) and (checks_given >= 1 or near_king)
    if (winning and checks_given >= ATTACK_CHECKS) or opened:
        # The king is the target: what the checks open up is the point.
        return Idea(
            goal="attack",
            first=first,
            line=tuple(pv),
            blow=blow[1] if blow else None,
            blow_move=pv[blow[0]] if blow else None,
            blow_board=boards[blow[0]] if blow else None,
            blow_at=blow_at,
            preparation=preparation,
            starts_with=starts_with,
            target=board.king(not us),
            flight=_flight(board, not us),
            checks=checks,
        )
    # A won position with nothing to win in sight is won by technique, never "no tactic
    # here -- gain space on the flank" in a king-and-pawn ending (Lichess pawnEndgame,
    # zugzwang and quietMove puzzles, 2026-09-23): the trade into a won ending, the king
    # leading, a pawn breakthrough.
    if winning:
        technique = _technique(board, boards, pv)
        if technique is not None:
            return replace(technique, starts_with=starts_with, checks=checks)
    # Material given and taken back by a tactic, the position better for it: a
    # combination, never "no tactic here" (an exchange sacrifice and a skewer that wins
    # the rook back was read as a rook behind a passed pawn, Lichess 03gJT).
    given = _sacrifices(boards, pv, min(len(pv), 8))
    primary = blow is not None and blow[1].name in PRIMARY
    if given and primary and analysis.lines[0].expected_score >= 0.6:
        return Idea(
            goal="combination",
            first=first,
            line=tuple(pv),
            gain=gain,
            blow=blow[1] if blow else None,
            blow_move=pv[blow[0]] if blow else None,
            blow_board=boards[blow[0]] if blow else None,
            blow_at=blow_at,
            preparation=preparation,
            starts_with=starts_with,
            checks=checks,
        )
    threat, threatened, follow = _threat(board, analysis.threats, first)
    if threat is not None:
        return Idea(
            goal="defend",
            first=first,
            line=tuple(pv),
            starts_with=starts_with,
            threat=threat,
            threatened=threatened,
            prepared=follow is not None,
            follow=follow,
            checks=checks,
        )
    return Idea(goal="plan", first=first, line=tuple(pv), starts_with=starts_with, checks=checks)


def _ignores(board: chess.Board, best: chess.Move, threat: chess.Move) -> bool:
    """The best move lets their direct threat come: it gives check (the attack goes on, and
    the capture waits), or after it the capture is still there and still wins. Black's
    14...Bh2+ in Capablanca-Marshall, with hxg4 hanging over the knight, was "your opponent
    threatens to win material: meet it first"."""
    if board.gives_check(best):
        return True
    after = board.copy(stack=False)
    after.push(best)
    return threat in after.legal_moves and exchange.exchange(after, threat) > 0


def ending_of(board: chess.Board) -> bool:
    """An ending: four pieces or fewer left besides kings and pawns."""
    return _pieces_left(board) <= 4


def _pieces_left(board: chess.Board) -> int:
    """Pieces other than kings and pawns, both sides."""
    return sum(
        1 for p in board.piece_map().values() if p.piece_type not in (chess.KING, chess.PAWN)
    )


def _ending_name(board: chess.Board) -> str:
    """The kind of ending on `board`: "pawn", "rook", "queen", "bishop", "knight", or
    "minor-piece" / "mixed"."""
    kinds = {p.piece_type for p in board.piece_map().values()} - {chess.KING, chess.PAWN}
    if not kinds:
        return "pawn"
    if len(kinds) == 1:
        return _NAMES[kinds.pop()]
    if kinds <= {chess.BISHOP, chess.KNIGHT}:
        return "minor-piece"
    return "mixed"


def _technique(
    board: chess.Board, boards: Sequence[chess.Board], pv: Sequence[chess.Move]
) -> Idea | None:
    """How a won position with no material in sight is won: trading into a won ending
    (the player's first moves take pieces off, material stays), the king leading in an
    ending (with the opposition when the kings stand face to face, one square apart, and
    they must move), or a pawn breakthrough in a pawn ending (a pawn given so another runs
    to the seventh). None when the line is none of these."""
    us = board.turn
    first = pv[0]
    start_pieces = _pieces_left(board)
    later = boards[min(len(boards) - 1, 4)]
    level_material = _balance(later, us) >= _balance(board, us) - 1
    # Pieces coming off in the first two moves each: Rh2+ Kd3 Rxc2 Kxc2 trades too.
    trades = any(boards[j].is_capture(pv[j]) for j in range(min(len(pv), 4)))
    if start_pieces and _pieces_left(later) < start_pieces and level_material and trades:
        return Idea(goal="convert", first=first, line=tuple(pv), ending=_ending_name(later))
    endgame = start_pieces <= 2 and not board.pieces(chess.QUEEN, chess.WHITE) | board.pieces(
        chess.QUEEN, chess.BLACK
    )
    mover = board.piece_type_at(first.from_square)
    if endgame and mover == chess.KING:
        after = boards[1]
        mine, theirs = after.king(us), after.king(not us)
        opposition = False
        if mine is not None and theirs is not None:
            df = abs(chess.square_file(mine) - chess.square_file(theirs))
            dr = abs(chess.square_rank(mine) - chess.square_rank(theirs))
            opposition = (df == 0 and dr == 2) or (dr == 0 and df == 2)
        return Idea(goal="king", first=first, line=tuple(pv), opposition=opposition)
    if start_pieces == 0 and mover == chess.PAWN:
        seventh = 6 if us == chess.WHITE else 1
        for j in range(2, min(len(pv), HORIZON), 2):
            move = pv[j]
            if boards[j].piece_type_at(move.from_square) == chess.PAWN and (
                chess.square_rank(move.to_square) == seventh or move.promotion
            ):
                given = boards[1].is_capture(pv[1]) if len(pv) > 1 else False
                if given and pv[1].to_square == first.to_square:
                    return Idea(
                        goal="breakthrough", first=first, line=tuple(pv), runner=move.from_square
                    )
                break
    return None


# The tactics worth naming beside the blow: the ones a player learns as patterns.
ALSO = ("fork", "skewer", "discoveredAttack", "doubleCheck", "pin", "absolutePin", "trappedPiece")


def _also(idea: Idea, boards: Sequence[chess.Board], pv: Sequence[chess.Move]) -> str | None:
    """A second tactic of a different kind that the line also cashes in, on the player's
    moves up to the blow: named beside it, never instead of it."""
    if idea.blow is None or idea.goal not in ("win", "promote", "combination", "attack"):
        return None
    upto = 2 * (idea.blow_at - 1)
    main = "pin" if idea.blow.name == "absolutePin" else idea.blow.name
    for i in range(0, upto + 1, 2):
        if i >= len(pv):
            break
        for tag in motifs.tags_for_move(boards[i], pv[i], pv=list(pv[i + 1 :])):
            name = "pin" if tag.name == "absolutePin" else tag.name
            if name in ALSO and name != main and motifs.pays_off(boards, pv, i, tag):
                return tag.name
    return None


def _pinned_guard(
    idea: Idea, boards: Sequence[chess.Board], pv: Sequence[chess.Move]
) -> chess.Square | None:
    """A piece of theirs that bears on the square the blow (or the first move) lands on but
    cannot take, being pinned to its own king -- and whose capture would otherwise have
    won: the reason the knight on f3 can fork with a pawn beside it."""
    if idea.blow_move is not None and idea.blow_board is not None:
        before, move = idea.blow_board, idea.blow_move
    else:
        before, move = boards[0], pv[0]
    after = before.copy(stack=False)
    after.push(move)
    mine = after.piece_at(move.to_square)
    if mine is None:
        return None
    for guard in after.attackers(not mine.color, move.to_square):
        kind = after.piece_type_at(guard)
        if kind is None or kind == chess.KING or move.to_square in after.pin(not mine.color, guard):
            continue
        cheaper = VALUES.get(kind, 0) < VALUES.get(mine.piece_type, 0)
        if cheaper or not after.attackers(mine.color, move.to_square):
            return guard
    return None


def _intermezzo(boards: Sequence[chess.Board], pv: Sequence[chess.Move], upto: int) -> bool:
    """The player, after one of theirs is taken, first plays a check elsewhere and only
    then takes back on that square: an in-between move."""
    us = boards[0].turn
    for j in range(2, upto - 2, 2):
        taken = pv[j - 1]
        victim = boards[j - 1].piece_at(taken.to_square)
        if victim is None or victim.color != us:
            continue
        move = pv[j]
        if move.to_square == taken.to_square or not boards[j].gives_check(move):
            continue
        if pv[j + 2].to_square == taken.to_square and boards[j + 2].is_capture(pv[j + 2]):
            return True
    return False


def _retaken(board: chess.Board, square: chess.Square) -> bool:
    """The side to move on `board` can legally take on `square`: a recapture that exists,
    not one a pinned piece or a king walking into a guarded square would make."""
    return any(m.to_square == square for m in board.legal_moves)


# How much worse the next-best move must be for a draw to be the point: the draw saves the game.
RESCUE = 0.15


def _drawn(
    boards: Sequence[chess.Board],
    pv: Sequence[chess.Move],
    score: float,
    second: float | None = None,
) -> str | None:
    """How the line saves the game, when it is a draw the player forces: stalemate on the
    board, check after check with the game level (a perpetual), or a position that comes
    round again. None when the line is not that -- or when the draw saves nothing: the
    next-best move (`second`, its expected score) does as well, and a level line that goes
    round in a circle is the engine marking time, not a rescue ("you can hold the game" was
    said at 80 of the master games' 93 such moves, all level whatever was played,
    2026-09-24)."""
    if not 0.35 <= score <= 0.65:
        return None
    if second is not None and second > score - RESCUE:
        return None
    if any(b.is_stalemate() for b in boards[1:]):
        return "stalemate"
    ours = [j for j in range(0, min(len(pv), 10), 2)]
    if len(ours) >= 3 and all(boards[j].gives_check(pv[j]) for j in ours):
        return "perpetual"
    seen: set[str] = set()
    for b in boards:
        key = b.epd()
        if key in seen:
            return "repetition"
        seen.add(key)
    return None


def _sacrifices(
    boards: Sequence[chess.Board], pv: Sequence[chess.Move], upto: int
) -> tuple[tuple[int, str], ...]:
    """The player's sacrifices in `pv[:upto]`, and what each gives. Counted against the
    material the line starts with: after the reply the player is two points or more
    down, the player's next move does not make it all back, and it goes deeper than the
    sacrifice before. So a bishop for two pawns is one, and a rook given in two steps
    (Rxf7+ Rxf7 Rxf7+ Kxf7, the first rook back at once) is one; a rook given for a
    queen taken the next move, or a pawn the bishop takes and a pawn takes back, is not
    (both were said to "give up material", 2026-09-23)."""
    us = boards[0].turn
    start = _balance(boards[0], us)
    lowest = start - 1  # a new sacrifice must go below this
    out: list[tuple[int, str]] = []
    for i in range(0, min(upto, len(pv)), 2):
        if i + 2 >= len(boards):
            break
        after_reply = _balance(boards[i + 2], us)
        if after_reply > start - 2 or after_reply >= lowest:
            continue
        if i + 3 < len(boards) and _balance(boards[i + 3], us) >= start:
            continue  # the next move makes it all back: a trade
        reply = pv[i + 1]
        given = boards[i + 1].piece_at(reply.to_square)
        if given is None or given.color != us:
            continue
        lowest = after_reply
        took = boards[i].piece_type_at(pv[i].to_square) if boards[i].is_capture(pv[i]) else None
        if given.piece_type == chess.ROOK and took in (chess.BISHOP, chess.KNIGHT):
            out.append((i, "exchange"))
        else:
            out.append((i, _NAMES[given.piece_type]))
    return tuple(out)


def _first_gift(boards: Sequence[chess.Board], pv: Sequence[chess.Move], reach: int) -> int:
    """The plies of the line before the player takes a piece the opponent just put where it
    could be taken, worth more than what it took on the way there; `reach` if none."""
    for k in range(1, min(reach, len(pv) - 1), 2):
        move, answer = pv[k], pv[k + 1]
        if answer.to_square != move.to_square or not boards[k + 1].is_capture(answer):
            continue
        piece = boards[k].piece_type_at(move.from_square)
        took = boards[k].piece_type_at(move.to_square) if boards[k].is_capture(move) else None
        if VALUES.get(piece or chess.KING, 0) > VALUES.get(took or chess.KING, 0):
            return k + 1
    return reach


def line_gain(
    boards: Sequence[chess.Board], pv: Sequence[chess.Move], horizon: int
) -> tuple[int, int]:
    """What the line wins in its first `horizon` plies, in pawns, and the ply it stands at:
    the best the player's material stands after their opponent's replies, a queen of their
    own making not counted (making one is a promotion, not a win of "their queen"). A line
    that stops on the player's capture counts it less what can take back: "dxe4+ Ke3 Rxd4"
    is a pawn, not a rook, with Rxd4 to come; a puzzle's solution stops the same way."""
    board = boards[0]
    us = board.turn
    base = _balance(board, us)
    made = _promoted(boards, pv, us)
    gain, at = 0, 0
    for k in range(2, horizon + 1, 2):
        g = _balance(boards[k], us) - base - made[k]
        if g > gain:
            gain, at = g, k
    if horizon == len(pv) and horizon % 2 == 1:
        g = _balance(boards[horizon], us) - base - made[horizon]
        last = pv[horizon - 1]
        if _retaken(boards[horizon], last.to_square):
            g -= VALUES.get(boards[horizon].piece_type_at(last.to_square) or chess.PAWN, 0)
        if g > gain:
            gain, at = g, horizon
    return gain, at


def _legal_prefix(board: chess.Board, moves: Sequence[str]) -> list[chess.Move]:
    out: list[chess.Move] = []
    at = board.copy(stack=False)
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in at.legal_moves:
            break
        out.append(move)
        at.push(move)
    return out


def _balance(board: chess.Board, us: chess.Color) -> int:
    return sum(
        VALUES.get(piece.piece_type, 0) * (1 if piece.color == us else -1)
        for piece in board.piece_map().values()
    )


def _start(board: chess.Board, boards: Sequence[chess.Board], pv: Sequence[chess.Move]) -> str:
    """How the line starts: with a check, a promotion, a sacrifice (material given that the
    reply keeps), a capture, or quietly."""
    first = pv[0]
    if board.gives_check(first):
        kind = "check"
    elif first.promotion:
        kind = "promotion"
    elif board.is_capture(first):
        kind = "capture"
    else:
        kind = "quiet"
    if _sacrifices(boards, pv, 1):
        return "sacrifice"
    return kind


def _blow(
    boards: Sequence[chess.Board],
    pv: Sequence[chess.Move],
    upto: int,
    mate: int | None,
    allowed: Sequence[str] | None = None,
    *,
    gained: frozenset[chess.PieceType] = frozenset(),
) -> tuple[tuple[int, MotifTag] | None, int, tuple[str, ...]]:
    """The most telling tactic of the player's moves in `pv[:upto]`, the later one of two
    alike (the point, not the preparation); which of the player's moves carries it; and what
    the moves before it do to prepare it. `gained`: the kinds of piece the line nets; a
    tactic whose own move takes one of them, worth more than what it aims at, is not the
    point (`_capture_is_the_point`)."""
    best: tuple[tuple[int, int], int, MotifTag] | None = None
    tagged: dict[int, list[MotifTag]] = {}
    for i in range(0, max(upto, 1), 2):
        if i >= len(pv):
            break
        mate_in = mate - i // 2 if mate is not None else None
        tags = motifs.tags_for_move(boards[i], pv[i], pv=list(pv[i + 1 :]), mate_in=mate_in)
        tagged[i] = tags
        for tag in tags:
            rank = _BLOW_INDEX.get(tag.name)
            if rank is None or (allowed is not None and tag.name not in allowed):
                continue
            targeted = tag.name in motifs.TARGETED
            if mate is None and targeted and not motifs.pays_off(boards, pv, i, tag):
                continue
            if mate is None and targeted and _capture_is_the_point(boards[i], pv[i], tag, gained):
                continue
            if mate is None and tag.name in PREPARATION and i + 2 >= upto:
                # A decoy or a deflection prepares what a later move cashes in: on the last
                # of the player's moves before the win it explains what comes after it (the
                # rook that takes the queen "lures" the king that takes it back).
                continue
            key = (rank, -i)
            if best is None or key < best[0]:
                best = (key, i, tag)
    if best is None:
        return None, 1, ()
    _, ply, tag = best
    preparation: list[str] = []
    for i in range(0, ply + 1, 2):
        for other in tagged.get(i, []):
            if other.name in PREPARATION and other.name not in preparation:
                preparation.append(other.name)
    return (ply, tag), ply // 2 + 1, tuple(preparation)


def _capture_is_the_point(
    board: chess.Board, move: chess.Move, tag: MotifTag, gained: frozenset[chess.PieceType]
) -> bool:
    """Whether the move takes more than the tactic it carries aims at, and what it takes is
    what the line wins (`gained`, the kinds it nets): a capture of a queen that also forks a
    bishop and a pawn wins the queen by taking it, not by the fork ("That's a fork" beside a
    queen won on c2 whose targets were c1, h2 and a4: 35...Nxf5 in Johner-Nimzowitsch). A
    queen taken back is no win, and the pin that nets the rook after it stays the point.
    Kings are not counted: a check is no material target."""
    if not board.is_capture(move):
        return False
    taken = board.piece_type_at(move.to_square) or chess.PAWN  # en passant: a pawn
    if taken not in gained:
        return False
    after = board.copy(stack=False)
    after.push(move)
    aims = max(
        (
            VALUES.get(piece.piece_type, 0)
            for name in tag.squares
            if (square := chess.parse_square(name)) not in (move.from_square, move.to_square)
            and (piece := after.piece_at(square)) is not None
            and piece.color == after.turn
            and piece.piece_type != chess.KING
        ),
        default=0,
    )
    return VALUES.get(taken, 0) > aims


def _gained_kinds(
    board: chess.Board, end: chess.Board, us: chess.Color
) -> frozenset[chess.PieceType]:
    """The kinds of piece the line nets from `board` to `end`: taken from them more than
    they took of the player's."""
    them = not us
    return frozenset(
        kind
        for kind in _COUNTED
        if (len(board.pieces(kind, them)) - len(end.pieces(kind, them)))
        - (len(board.pieces(kind, us)) - len(end.pieces(kind, us)))
        > 0
    )


def _first_reaching(
    boards: Sequence[chess.Board], pv: Sequence[chess.Move], upto: int, target: int
) -> int:
    """The first ply, to `upto`, at which the line has won `target` or more."""
    for k in range(2, upto + 1):
        if k % 2 and k != upto:
            continue
        if line_gain(boards, pv, k)[0] >= target:
            return k
    return 0


def _promoted(
    boards: Sequence[chess.Board], pv: Sequence[chess.Move], us: chess.Color
) -> list[int]:
    """Per ply, what the player's new pieces add to the balance beyond the pawns they were:
    a queen made and still standing counts eight; one taken back, nothing."""
    out = [0]
    alive: dict[chess.Square, int] = {}
    for j, move in enumerate(pv):
        mover = boards[j].turn
        if mover == us:
            extra = alive.pop(move.from_square, None)
            if move.promotion:
                alive[move.to_square] = VALUES.get(move.promotion, 1) - 1
            elif extra is not None:
                alive[move.to_square] = extra
        else:
            alive.pop(move.to_square, None)
        out.append(sum(alive.values()))
    return out


def _won(
    board: chess.Board,
    end: chess.Board,
    us: chess.Color,
    played: Sequence[chess.Move],
    boards: Sequence[chess.Board],
    owed: dict[int, int] | None = None,
) -> tuple[str | None, chess.Square | None]:
    """What the line wins, in words ("their queen", "the exchange", "a rook for a knight"),
    and where the most valuable piece it wins stands now; with `owed`, net of what the
    exchange in progress took from the player."""
    them = not us
    nets = {
        kind: (len(board.pieces(kind, them)) - len(end.pieces(kind, them)))
        - (len(board.pieces(kind, us)) - len(end.pieces(kind, us)))
        - (owed or {}).get(kind, 0)
        for kind in _COUNTED
    }
    if owed and nets[chess.KNIGHT] * nets[chess.BISHOP] < 0:
        # Net of what they took, a knight for a bishop is a trade: "two pawns", not "a knight
        # and two pawns for a bishop".
        both = min(abs(nets[chess.KNIGHT]), abs(nets[chess.BISHOP]))
        for kind in (chess.KNIGHT, chess.BISHOP):
            nets[kind] -= both if nets[kind] > 0 else -both
    gained = {kind: n for kind, n in nets.items() if n > 0}
    given = {kind: -n for kind, n in nets.items() if n < 0}
    if not gained:
        return None, None
    top = max(gained, key=lambda kind: VALUES[kind])
    # Where the most valuable piece won stands now: the first of its kind the player takes.
    target: chess.Square | None = None
    for i, move in enumerate(played):
        if i % 2:
            continue
        taken = boards[i].piece_at(move.to_square)
        if taken is not None and taken.color == them and taken.piece_type == top:
            target = origin(board, boards, i, move.to_square)
            break
    return _won_words(gained, given), target


def _line_end(boards: Sequence[chess.Board], pv: Sequence[chess.Move], at: int) -> chess.Board:
    """The board `at` plies into the line; a line that stops on the player's capture loses
    the capturing piece where it can be taken back, as `line_gain` counts it ("you win the
    exchange", not "a rook", after Nxe1)."""
    end = boards[at]
    if at == len(pv) and at % 2 == 1 and _retaken(end, pv[at - 1].to_square):
        end = end.copy(stack=False)
        end.remove_piece_at(pv[at - 1].to_square)
    return end


def _top_taken(board: chess.Board, end: chess.Board, us: chess.Color) -> str | None:
    """The most valuable kind of the opponent's pieces the line takes, in words: "their
    queen", "a rook", "a knight"."""
    them = not us
    taken = [k for k in _COUNTED if len(end.pieces(k, them)) < len(board.pieces(k, them))]
    if not taken:
        return None
    top = max(taken, key=lambda kind: VALUES[kind])
    return "their queen" if top == chess.QUEEN else f"a {_NAMES[top]}"


def origin(
    board: chess.Board, boards: Sequence[chess.Board], ply: int, square: chess.Square
) -> chess.Square | None:
    """Where the piece standing on `square` at `ply` stood at the start, followed back."""
    at = square
    for i in range(ply - 1, -1, -1):
        before, after = boards[i], boards[i + 1]
        moved = [
            sq
            for sq in chess.SQUARES
            if before.piece_at(sq) != after.piece_at(sq) and after.piece_at(sq) is None
        ]
        if after.piece_at(at) is not None and before.piece_at(at) != after.piece_at(at):
            # The piece arrived on `at` at this ply: it came from the square it left.
            arrived = after.piece_at(at)
            came = [sq for sq in moved if before.piece_at(sq) == arrived]
            if not came and arrived is not None:
                # A promotion: the new piece came from the pawn that left ("their rook on
                # d1" was named for the pawn on d2 that underpromotes in the line).
                came = [
                    sq
                    for sq in moved
                    if (p := before.piece_at(sq)) is not None
                    and p.color == arrived.color
                    and p.piece_type == chess.PAWN
                ]
            if came:
                at = came[0]
    return at if board.piece_at(at) is not None else None


_NAMES = {
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
    chess.PAWN: "pawn",
}
_PLURALS = {
    "queen": "queens",
    "rook": "rooks",
    "bishop": "bishops",
    "knight": "knights",
    "pawn": "pawns",
}
_NUMBERS = {1: "a", 2: "two", 3: "three", 4: "four"}


def _pieces(counts: dict[int, int]) -> str:
    parts = []
    for kind in _COUNTED:
        n = counts.get(kind, 0)
        if n == 1:
            parts.append(f"a {_NAMES[kind]}")
        elif n > 1:
            parts.append(f"{_NUMBERS.get(n, str(n))} {_PLURALS[_NAMES[kind]]}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _won_words(gained: dict[int, int], given: dict[int, int]) -> str:
    if given:
        minors = {chess.KNIGHT, chess.BISHOP}
        if gained == {chess.ROOK: 1} and sum(given.values()) == 1 and set(given) <= minors:
            return "the exchange"
        return f"{_pieces(gained)} for {_pieces(given)}"
    if gained == {chess.QUEEN: 1}:
        return "their queen"
    return _pieces(gained)


def _flight(board: chess.Board, colour: chess.Color) -> tuple[chess.Square, ...]:
    """The squares `colour`'s king could step to, were it their move: not held by their own
    pieces, not attacked by the other side."""
    king = board.king(colour)
    if king is None:
        return ()
    view = board.copy(stack=False)
    view.remove_piece_at(king)  # a square behind the king on a checking line is no refuge
    out = []
    for square in chess.SQUARES:
        if chess.square_distance(square, king) != 1:
            continue
        piece = board.piece_at(square)
        if piece is not None and piece.color == colour:
            continue
        if view.is_attacked_by(not colour, square):
            continue
        out.append(square)
    return tuple(out)


# A prepared threat is read only when what it prepares is mate or at least a minor piece:
# every move attacks something, and "your opponent is preparing to win a pawn" would be
# said of half the moves of a quiet game.
PREPARED_MIN = 3


def _threat(
    board: chess.Board, threats: Sequence[str], best: chess.Move | None = None
) -> tuple[chess.Move | None, str | None, chess.Move | None]:
    """The opponent's threat, if the player passed, when it mates or wins material; what it
    threatens ("mate", or "your rook"); and, when the threat is a quiet move that prepares it
    (their best reply to a pass threatens mate or a piece next), the move that would follow.
    A prepared threat counts only when the best move takes it away: where the best move lets
    it come and strikes first (g4 while ...b4 still wins the knight), the point is the counter,
    and "stop it before it comes" would be false (the master games' notes, 2026-09-24)."""
    if not threats or board.is_check():
        return None, None, None
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    for uci in threats:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if move not in passed.legal_moves:
            continue
        after = passed.copy(stack=False)
        after.push(move)
        if after.is_checkmate():
            return move, "mate", None
        taken = passed.piece_at(move.to_square)
        # A capture that wins material by the exchange, legal recaptures and all: a knight
        # left to a rook with only a pinned piece to take back is a threat (2026-09-23).
        if taken is not None and taken.color == board.turn and exchange.exchange(passed, move) > 0:
            if best is not None and best in board.legal_moves and _ignores(board, best, move):
                continue  # the best move lets it come: the point is elsewhere
            return move, f"your {_NAMES[taken.piece_type]}", None
    # The engine's first reply to a pass, when quiet: what it would threaten next.
    for uci in threats[:1]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if move not in passed.legal_moves or passed.is_capture(move):
            continue
        prepared = plans.threat(passed, move)
        if prepared is None or prepared.kind == "trade":
            continue
        if best is not None and best in board.legal_moves:
            answered = board.copy(stack=False)
            answered.push(best)
            if move in answered.legal_moves:
                # The same threat, on the same target: after Nd4 the rook's Rd1 goes for the
                # knight, not the queen it went for, and the queen's threat is met.
                still = plans.threat(answered, move)
                if (
                    still is not None
                    and still.kind == prepared.kind
                    and (still.kind == "mate" or still.square == prepared.square)
                ):
                    continue
        if prepared.kind == "mate":
            return move, "mate", prepared.follow
        target = passed.piece_at(prepared.square)
        if target is not None and prepared.gain >= PREPARED_MIN:
            return move, f"your {_NAMES[target.piece_type]}", prepared.follow
    return None, None, None
