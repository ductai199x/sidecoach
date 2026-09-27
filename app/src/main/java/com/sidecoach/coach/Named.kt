package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.sanLine
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.Move
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareName
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Line

/**
 * The engine on what the player asks about, so the coach analyses it rather than guesses.
 * A port of `sidecoach/coach/named.py`.
 *
 * "What about Nf3?" once got "not among the engine's best moves; it prefers Bb5" -- true, and
 * all the object had. The player may name one move, a whole line, several alternatives, a
 * move of the opponent's, or an idea in words. So the examination is agentic: the model
 * decides what to examine (a list of probes), the engine examines each, and the results go
 * into the object before the coach answers. Every number is the engine's; every effect is
 * computed on the board. Below LINE the engine's own preferred move is not named.
 */

const val PLAN_SYSTEM =
    "A chess player, in the middle of a game, asks their coach about a move, a line, or an " +
        "idea. You turn their words into what the engine should examine: a list of probes, each a " +
        "sequence of moves in standard notation from the position given, played in order, " +
        "starting with the side to move. If they ask what the opponent might do, the probe " +
        "starts with the word 'pass' (the player waits) and then the opponent's move; if they " +
        "say their own move first and then the opponent's ('after I castle, he plays d5'), the " +
        "probe is their move and then the opponent's, in that order. An idea in " +
        "words ('trade queens', 'take on f7', 'castle') becomes the moves that do it. Several " +
        "alternatives are several probes, at most four. Label each probe in the player's own " +
        "terms ('your Nf3', 'the line Bb5 a6 Ba4', 'if they take on e5'). A question about a " +
        "piece or a square rather than a move ('is my knight safe?', 'what does the bishop on " +
        "d3 do?', 'what about e5?') is a probe with a square instead of moves. If their words " +
        "name nothing to examine, return an empty list. Answer in JSON only: " +
        "{\"probes\": [{\"label\": str, \"moves\": [str]} | {\"label\": str, \"square\": str}]}"

private val JSON_BLOCK = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL)

/** A move as a player types it, with or without a move number; "pass" is the player waiting. */
private val TOKEN = Regex(
    "(?<![A-Za-z0-9])(?:\\d{1,3}\\.(?:\\.\\.)?\\s*)?((?:[a-h][1-8][a-h][1-8][qrbn]?" +
        "|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x?[a-h][1-8]|[a-h][1-8]|O-O-O|O-O|0-0-0|0-0)" +
        "(?:=[QRBN])?[+#]?)(?![A-Za-z0-9])",
)
private val UCI_RE = Regex("^[a-h][1-8][a-h][1-8][qrbn]?$")

/**
 * A piece named with its square ("the white bishop on d3", "my knight at f6") or a bare square
 * asked about ("what about e5", "square d5").
 */
private val PIECE_ON = Regex(
    "(?i)(?:(?<colour>white|black|my|your|their|his|her)\\s+)?" +
        "(?<piece>king|queen|rook|bishop|knight|pawn)s?\\s+(?:on|at)\\s+(?<square>[a-h][1-8])\\b",
)
private val SQUARE_ASKED = Regex("(?i)\\b(?:square|on|about|at)\\s+(?<square>[a-h][1-8])\\b")

/** What one examination needs of an engine: the lines for a position. */
fun interface EngineLike {
    fun analyse(board: Board, depth: Int?): List<Line>
}

/**
 * One thing to examine: moves from the position, in the player's terms -- or a piece or
 * square, when that is what they asked about.
 */
class Probe(
    val label: String,
    /** SAN, or "pass" for a null move (the opponent's move first). */
    val moves: List<String>,
    val parsed: MutableList<Move> = ArrayList(),
    /** The first move that could not be played, if any. */
    var illegal: String? = null,
    /** A piece or square probe instead of moves. */
    val square: Int? = null,
)

/**
 * The probe's moves played on the board in order; an illegal one ends the probe and is
 * remembered, so the coach can say so.
 */
fun parseProbe(label: String, moves: List<String>, board: Board): Probe {
    val probe = Probe(label = label, moves = ArrayList(moves))
    val replay = board.copy(withStack = false)
    for (raw in moves) {
        val token = raw.trim().replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        val move: Move
        if (token.lowercase() == "pass") {
            if (replay.isCheck()) {
                probe.illegal = "pass (in check)"
                break
            }
            move = Move.NULL
        } else {
            val parsed = parseToken(replay, token)
            if (parsed == null) {
                probe.illegal = token
                break
            }
            move = parsed
        }
        probe.parsed.add(move)
        replay.push(move)
    }
    return probe
}

/**
 * Without a planner: the moves named in the words, as one line, as far as legal; or the pieces
 * and squares named, one probe each.
 */
fun probesByWords(text: String, board: Board): List<Probe> {
    val pieces = piecesByWords(text, board)
    if (pieces.isNotEmpty()) return pieces
    val moves = ArrayList<String>()
    val replay = board.copy(withStack = false)
    for (match in TOKEN.findAll(text)) {
        val token = match.groupValues[1].replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        val move = parseToken(replay, token)
        if (move == null) {
            if (moves.isEmpty()) continue
            break
        }
        moves.add(replay.san(move))
        replay.push(move)
    }
    if (moves.isEmpty()) return emptyList()
    val label = sanLine(board, replayMoves(board, moves).map { it.uci() })
    return listOf(parseProbe(label, moves, board))
}

/** The pieces named with their square, or the squares asked about, as probes. */
fun piecesByWords(text: String, board: Board): List<Probe> {
    val out = ArrayList<Probe>()
    val seen = HashSet<Int>()
    for (m in PIECE_ON.findAll(text)) {
        val square = parseSquare(m.groups["square"]!!.value.lowercase())
        if (!seen.add(square)) continue
        out.add(Probe(label = m.value.trim(), moves = emptyList(), square = square))
    }
    if (out.isNotEmpty()) return out.take(4)
    for (m in SQUARE_ASKED.findAll(text)) {
        val square = parseSquare(m.groups["square"]!!.value.lowercase())
        if (!seen.add(square)) continue
        out.add(Probe(label = squareName(square), moves = emptyList(), square = square))
    }
    return out.take(4)
}

/**
 * The piece on `at` as it controls `square`: marked pinned when it is pinned to its king off the
 * line through `square` -- it still keeps the enemy king off, but takes nothing that lands there.
 */
private fun controller(board: Board, at: Int, square: Int): String {
    val colour = board.colourAt(at)
    if (colour != null && board.pinMask(colour, at) and BB_SQUARES[square] == BB_EMPTY) {
        return "${nameOf(board, at)} (pinned)"
    }
    return nameOf(board, at)
}

/**
 * The board on one piece or square: what stands there, where it can go, what it attacks and
 * defends, what attacks and defends it, whether it is pinned. Static facts of the position, no
 * search. Attack and defence are legal captures and recaptures ([Exchange]): a piece pinned to
 * its king attacks and defends nothing off its line, and a king does not take back where the
 * other side still guards.
 */
fun examinePiece(probe: Probe, board: Board, userColour: Boolean): String {
    val square = probe.square!!
    val head = "Engine on ${probe.label}, which the player asked about:"
    val piece = board.pieceAt(square)
    if (piece == null) {
        val holders = board.attackers(userColour, square).sorted()
        val theirs = board.attackers(!userColour, square).sorted()
        val parts = arrayListOf("${squareName(square)} is empty")
        if (holders.isNotEmpty()) {
            parts.add(
                "the player controls it with the " +
                    holders.take(3).joinToString(" and the ") { controller(board, it, square) },
            )
        }
        if (theirs.isNotEmpty()) {
            parts.add(
                "the opponent controls it with the " +
                    theirs.take(3).joinToString(" and the ") { controller(board, it, square) },
            )
        }
        if (holders.isEmpty() && theirs.isEmpty()) parts.add("nobody controls it")
        return "$head " + parts.joinToString("; ") + "."
    }
    val whose = if (piece.colour == userColour) "the player's" else "the opponent's"
    val pieceName = "$whose ${COACH_PIECE_NAMES[piece.pieceType]} on ${squareName(square)}"
    val parts = arrayListOf("$pieceName stands there")
    // Where it can go, on the board as it stands, with its own side to move. Never onto the
    // enemy king: with the turn handed to the side giving check, the generator offers it.
    val view = board.copy(withStack = false)
    view.turn = piece.colour
    val moves = view.legalMoves().filter { it.from == square && view.pieceTypeAt(it.to) != KING }
    if (moves.isNotEmpty()) {
        parts.add("it can go to " + moves.map { squareName(it.to) }.toSortedSet().joinToString(", "))
    } else {
        parts.add("it has no legal move")
    }
    // It attacks what it could take among those moves, and the enemy king when it gives check
    // (a pinned piece still does); it defends the pieces it could legally take back on.
    val attacks = board.attacks(square).filter { t ->
        val o = board.pieceAt(t)
        o != null && o.colour != piece.colour && (o.pieceType == KING || moves.any { m -> m.to == t })
    }
    val defends = board.attacks(square).filter { t ->
        val o = board.pieceAt(t)
        o != null && o.colour == piece.colour && o.pieceType != KING &&
            square in Exchange.defenders(board, piece.colour, t)
    }
    if (attacks.isNotEmpty()) {
        parts.add("it attacks the " + attacks.take(4).joinToString(" and the ") { nameOf(board, it) })
    }
    if (defends.isNotEmpty()) {
        parts.add("it defends the " + defends.take(4).joinToString(" and the ") { nameOf(board, it) })
    }
    // A king is attacked by what gives it check, pinned or not; any other piece by what can
    // legally take it.
    val attackers = if (piece.pieceType == KING) {
        board.attackers(!piece.colour, square).sorted()
    } else {
        Exchange.capturers(board, !piece.colour, square)
    }
    val defenders = Exchange.defenders(board, piece.colour, square)
    if (attackers.isNotEmpty()) {
        parts.add(
            "it is attacked by the " + attackers.take(3).joinToString(" and the ") { nameOf(board, it) },
        )
    } else {
        parts.add("nothing attacks it")
    }
    if (piece.pieceType != KING) {
        if (defenders.isNotEmpty()) {
            parts.add(
                "it is defended by the " +
                    defenders.take(3).joinToString(" and the ") { nameOf(board, it) },
            )
        } else {
            parts.add("nothing defends it")
        }
        val pin = pinTarget(board, square)
        if (pin != null) {
            parts.add(
                "it is pinned to the ${nameOf(board, pin.second)} by the ${nameOf(board, pin.first)}",
            )
        }
    }
    return "$head " + parts.joinToString("; ") + "."
}

/** A move as the player wrote it: SAN, or UCI ("e2e4"); null when illegal. */
private fun parseToken(board: Board, token: String): Move? {
    if (UCI_RE.matches(token.lowercase())) {
        val move = runCatching { Move.fromUci(token.lowercase()) }.getOrNull() ?: return null
        return if (board.isLegal(move)) move else null
    }
    return board.parseSan(token)
}

/** The moves of the probe as a line, or the raw tokens when a pass is among them. */
private fun playedText(probe: Probe, board: Board): String =
    if (probe.parsed.all { !it.isNull }) {
        sanLine(board, probe.parsed.filter { !it.isNull }.map { it.uci() })
    } else {
        probe.moves.take(probe.parsed.size).joinToString(" ")
    }

/**
 * Without an engine: the position the probe's moves lead to, and what the first move does. No
 * score; the FEN so the position can be read.
 */
fun examineStatic(probe: Probe, board: Board): String {
    val head = "Engine on ${probe.label}, which the player asked about:"
    if (probe.parsed.isEmpty()) return "$head ${probe.illegal} is not a legal move here."
    val after = board.copy(withStack = false)
    for (move in probe.parsed) after.push(move)
    val played = playedText(probe, board)
    val parts = arrayListOf(head)
    if (probe.illegal != null) parts.add("after $played, ${probe.illegal} is not a legal move;")
    var tail = "after $played the position is ${after.fen()}"
    if (after.isGameOver()) tail += " (the game is over: ${after.result()})"
    val first = probe.parsed.firstOrNull { !it.isNull }
    if (first != null && !probe.parsed[0].isNull) {
        val does = effectsText(board, first, hazards = true)
        if (does.isNotEmpty()) tail += "; what ${board.san(first)} does: $does"
    }
    parts.add(tail)
    return parts.joinToString(" ") + "."
}

private fun replayMoves(board: Board, sans: List<String>): List<Move> {
    val out = ArrayList<Move>()
    val replay = board.copy(withStack = false)
    for (san in sans) {
        val move = replay.parseSan(san) ?: break
        out.add(move)
        replay.push(move)
    }
    return out
}

/**
 * The engine's word on one probe, one line of text for the object. A probe that one of the
 * position's own lines ([known]) begins with is read from that line: a second, shallower search
 * after it answered Bd2 with Kh6 where the lines above had Rxh3, and the model, shown both, said
 * Bd2 avoided Rxh3 (the dev audit, 2026-09-26).
 */
fun examine(
    probe: Probe,
    board: Board,
    engine: EngineLike,
    best: Line?,
    userColour: Boolean,
    level: DisclosureLevel,
    depth: Int = 14,
    known: List<Line> = emptyList(),
): String {
    val head = "Engine on ${probe.label}, which the player asked about:"
    if (probe.parsed.isEmpty()) return "$head ${probe.illegal} is not a legal move here."
    val after = board.copy(withStack = false)
    for (move in probe.parsed) after.push(move)
    val played = playedText(probe, board)
    val parts = arrayListOf(head)
    if (probe.illegal != null) parts.add("after $played, ${probe.illegal} is not a legal move;")
    if (after.isGameOver()) {
        parts.add("the game is over after $played (${after.result()}).")
        return parts.joinToString(" ")
    }
    val moves = probe.parsed.map { it.uci() }
    val own = known.firstOrNull { line ->
        line.moves.size > moves.size && line.moves.subList(0, moves.size) == moves
    }
    val lines = if (own != null && probe.parsed.all { !it.isNull }) {
        listOf(
            Line(
                moves = own.moves.subList(moves.size, own.moves.size),
                depth = own.depth,
                // From the side to move after the probe, as a fresh search would give it.
                expectedScore = if (moves.size % 2 == 0) own.expectedScore else 1.0 - own.expectedScore,
            ),
        )
    } else {
        engine.analyse(after, depth)
    }
    if (lines.isEmpty()) {
        return "$head the engine has nothing to say about the position after $played."
    }
    val score = if (after.turn == userColour) {
        lines[0].expectedScore
    } else {
        1.0 - lines[0].expectedScore
    }
    // `f"{score:.2f}"`: never a comma, and a tie rounded half to even, whatever the device's
    // locale (the parity audit cc-02: "%.2f" read "0,53" to the model and the voice).
    var tail = "after $played the expected score is ${formatted(score, 2)} from the player's side"
    // Their move first ("if they take on b4") is the player passing, which is no move of theirs:
    // measured against the best it came out "after pass axb4 you are losing, a serious step down
    // from the best (the best is Bxc3)" (the simulated students).
    val passing = probe.parsed.isNotEmpty() && probe.parsed[0].isNull
    if (best != null && !passing) {
        var gap = best.expectedScore - score
        // The engine's own first move, searched again after it is played, comes back a hair
        // off its first score: "12. Nc3: ... a hair behind the best (the best is Nc3)" (the
        // simulated students, 2026-09-26). The best move is the best.
        if (best.moves.isNotEmpty() && probe.parsed.size == 1 && probe.parsed[0].uci() == best.moves[0]) {
            gap = 0.0
        }
        if (gap > 0.005) {
            tail += ", ${formatted(gap, 2)} behind the engine's best"
            if (level.value >= DisclosureLevel.LINE.value && best.moves.isNotEmpty()) {
                tail += " (${board.san(Move.fromUci(best.moves[0]))})"
            }
        } else if (gap < -0.005) {
            tail += ", better than the engine's first line"
        } else {
            tail += ", as good as the engine's best"
        }
    }
    tail += ";"
    parts.add(tail)
    val replyPlies = if (level.value >= DisclosureLevel.LINE.value) null else 2
    val reply = sanLine(after, lines[0].moves, limit = replyPlies)
    if (reply.isNotEmpty()) {
        val whose = if (after.turn != userColour) {
            "the opponent's reply would be"
        } else {
            "the engine then plays"
        }
        parts.add("$whose $reply;")
    }
    val first = probe.parsed.firstOrNull { !it.isNull }
    if (first != null && !probe.parsed[0].isNull) {
        val does = effectsText(board, first, hazards = true)
        if (does.isNotEmpty()) {
            parts.add("what ${board.san(first)} does: $does.")
            return parts.joinToString(" ")
        }
    }
    parts[parts.size - 1] = parts[parts.size - 1].trimEnd(';') + "."
    return parts.joinToString(" ")
}

private val ENGINE_ON = Regex(
    "^Engine on (?<label>.+?), which the player asked about: (?<body>.*)$",
    RegexOption.DOT_MATCHES_ALL,
)
private val PROBE_LINE = Regex("after (?<moves>.+?) the expected score is")
private val REPLY_LINE = Regex("(?:the opponent's reply would be|the engine then plays) (?<moves>[^;]+)")
private val MOVE_NUMBER = Regex("^\\d{1,3}\\.(?:\\.\\.)?")

/**
 * The positions along each examination: after every move of the probe, then after every move
 * of the engine's reply. The coach may recite what it was shown, so a move or a piece that is
 * real only further along that line is checked there, not faulted.
 */
fun disclosedFrames(board: Board, facts: List<String>): List<Board> {
    val frames = ArrayList<Board>()
    for (fact in facts) {
        val m = ENGINE_ON.matchEntire(fact) ?: continue
        val body = m.groups["body"]!!.value
        var replay = board.copy(withStack = false)
        val probe = PROBE_LINE.find(body)
        val tokens = ArrayList<String>()
        if (probe != null) tokens.addAll(probe.groups["moves"]!!.value.split(" ").filter { it.isNotEmpty() })
        val reply = REPLY_LINE.find(body)
        if (reply != null) {
            tokens.addAll(
                reply.groups["moves"]!!.value.trimEnd('.', ' ').split(" ").filter { it.isNotEmpty() },
            )
        }
        for (raw in tokens) {
            val token = MOVE_NUMBER.replace(raw, "").trim()
            if (token.isEmpty()) continue
            val move: Move
            if (token.lowercase() == "pass") {
                move = Move.NULL
            } else {
                move = replay.parseSan(token) ?: break
            }
            replay = replay.copy(withStack = false)
            replay.push(move)
            frames.add(replay)
        }
    }
    return frames
}

/**
 * The examination as the coach says it out loud, the same facts in plainer words. Numbers
 * become words; nothing is added.
 */
fun spoken(fact: String): String {
    val m = ENGINE_ON.matchEntire(fact) ?: return fact
    val label = m.groups["label"]!!.value
    var body = m.groups["body"]!!.value.trim()
    body = Regex("^after (.+?) the expected score is (\\d\\.\\d\\d) from the player's side")
        .replace(body) { mm ->
            if (mm.groupValues[1].startsWith("pass ")) {
                "if they got to play ${mm.groupValues[1].substring(5)} now, " +
                    standingWords(mm.groupValues[2].toDouble())
            } else {
                "after ${mm.groupValues[1]} ${standingWords(mm.groupValues[2].toDouble())}"
            }
        }
    body = Regex(", (\\d\\.\\d\\d) behind the engine's best(?: \\(([^)]+)\\))?")
        .replace(body) { mm ->
            ", " + gapWords(mm.groupValues[1].toDouble()) +
                (if (mm.groupValues[2].isNotEmpty()) " (the best is ${mm.groupValues[2]})" else "")
        }
    body = body.replace(", as good as the engine's best", ", as good as the best move")
        .replace(", better than the engine's first line", ", better than the engine's first choice")
    body = body.replace("the opponent's reply would be", "they would answer")
        .replace("the engine then plays", "then the engine plays")
    // Said aloud, a line is its first reply: "they would answer 9... Bxg2 10. Kxg2 Qd7 11. d5 Na6
    // 12. Nc3 O-O 13. a3 h6" came before every answer the AI then gave in words (the simulated
    // students' transcripts, 2026-09-26). The whole line stays in the fact. (The desktop makes
    // this substitution twice over; the second finds every line already cut, so once is the
    // same.)
    body = Regex("(they would answer|then the engine plays) ([^;]+)").replace(body) { mm ->
        "${mm.groupValues[1]} ${firstMoves(mm.groupValues[2], 2)}"
    }
    body = Regex("what (\\S+) does: ").replace(body) { mm -> mm.groupValues[1] + " " }
    // A piece or a square examined (examinePiece) is said to the player, not about them: "f3: the
    // player's knight on f3 stands there; it can go to ..." went out as it was written for the
    // model (a simulated student's session, 2026-09-26).
    body = body.replace("the player's", "your")
        .replace("the opponent's", "their")
        .replace("the player controls", "you control")
        .replace("the opponent controls", "they control")
    val piece = PIECE_STANDS.find(body)
    if (piece != null) {
        val (whose, name, square) = piece.destructured
        return "${whose.replaceFirstChar { it.uppercaseChar() }} $name on $square " +
            body.substring(piece.range.last + 1)
    }
    if (SQUARE_EMPTY.containsMatchIn(body)) return body // names its square itself
    return "$label: $body"
}

private val PIECE_STANDS = Regex("^(your|their) (\\w+) on ([a-h][1-8]) stands there; it ")
private val SQUARE_EMPTY = Regex("^[a-h][1-8] is empty")

private val MOVE_NUMBERED = Regex("^\\d+\\.")

/** The first [n] moves of a SAN line, with their move numbers: "9... Bxg2 10. Kxg2". */
private fun firstMoves(line: String, n: Int): String {
    val out = ArrayList<String>()
    var moves = 0
    for (token in line.split(Regex("\\s+")).filter { it.isNotEmpty() }) {
        out.add(token)
        if (!MOVE_NUMBERED.containsMatchIn(token)) {
            moves += 1
            if (moves >= n) break
        }
    }
    return out.joinToString(" ")
}

private fun gapWords(gap: Double): String = when {
    gap < 0.02 -> "a hair behind the best"
    gap < 0.05 -> "a little behind the best"
    gap < 0.15 -> "clearly behind the best"
    else -> "a serious step down from the best"
}

/**
 * The engine's word on what the player asked about: one line per probe, or none when their
 * words name nothing to examine. Without a planner the moves named in the words are examined
 * as one line, or the pieces and squares named each on their own.
 */
fun analyseNamed(
    text: String,
    board: Board,
    engine: EngineLike?,
    userColour: Boolean,
    level: DisclosureLevel,
    plannedProbes: List<Probe>? = null,
    depth: Int = 14,
    static: Boolean = false,
    piecesBoard: Board? = null,
    known: List<Line> = emptyList(),
): List<String> {
    var probes = plannedProbes ?: probesByWords(text, board)
    if ((board.turn != userColour && engine != null) || (engine == null && !static)) {
        // Only the position the player faces is theirs to ask lines about; the board's pieces
        // are theirs to ask about at any time.
        probes = probes.filter { it.square != null }
    }
    if (probes.isEmpty()) return emptyList()
    val out = ArrayList<String>()
    var best: Line? = null
    if (engine != null && probes.any { it.square == null }) {
        val before = known.toList().ifEmpty { engine.analyse(board, depth) }
        best = before.firstOrNull()
    }
    for (p in probes) {
        when {
            // A piece or square is read on the board as it stands ([piecesBoard]), even when the
            // question is about the position a move was played from: "b5 is empty" was said with
            // the player's bishop on b5 (the students' round 6).
            p.square != null -> out.add(examinePiece(p, piecesBoard ?: board, userColour))
            engine == null -> out.add(examineStatic(p, board))
            else -> out.add(examine(p, board, engine, best, userColour, level, depth, known = known))
        }
    }
    return out
}

/**
 * The message `plan_probes` sends the planner: the position, whose move and whose side the
 * player is, the legal moves, and their words.
 */
fun planRequest(text: String, board: Board, userColour: Boolean): String {
    val side = if (board.turn == com.sidecoach.chess.WHITE) "White" else "Black"
    val player = if (userColour == com.sidecoach.chess.WHITE) "White" else "Black"
    val legal = board.legalMoves().map { board.san(it) }.sorted().joinToString(" ")
    return "Position (FEN): ${board.fen()}\n$side to move. The player is $player.\n" +
        "Legal moves: $legal\n\nThe player says: \"${text.trim()}\"\n\nJSON only."
}

/**
 * The planner's JSON turned into probes on the board, with the words as the fallback. The call
 * to the model itself belongs to the caller, which owns the provider.
 */
fun probesFromPlan(json: String?, text: String, board: Board): List<Probe> {
    // `json.loads(_JSON.search(completion.text or "").group(0))`, anything unreadable {}: read
    // as Python reads it, strict, a null label None rather than "null" (the parity audit cc-05).
    val found = JSON_BLOCK.find(json ?: "")?.value
    val data = if (found != null) pyJsonOr(found, emptyMap<String, Any?>()) else emptyMap<String, Any?>()
    val probes = ArrayList<Probe>()
    val items = (data as? Map<*, *>)?.get("probes")
    for (item in if (pyTruthy(items)) pyIter(items) else emptyList()) {
        if (item !is Map<*, *>) continue
        val squareRaw = item["square"]
        val square = (if (pyTruthy(squareRaw)) pyStr(squareRaw) else "").trim().lowercase()
        val labelRaw = item["label"]
        if (square.isNotEmpty() && Regex("[a-h][1-8]").matches(square)) {
            val label = if (pyTruthy(labelRaw)) pyStr(labelRaw) else square
            probes.add(Probe(label = label, moves = emptyList(), square = parseSquare(square)))
        } else {
            // `item.get("moves", [])`: a null there is no list, and raises as Python's does.
            val movesRaw = if (item.containsKey("moves")) item["moves"] else emptyList<Any?>()
            val moves = pyIter(movesRaw).map { pyStr(it) }.filter { it.trim().isNotEmpty() }
            if (moves.isEmpty()) continue
            val label = if (pyTruthy(labelRaw)) pyStr(labelRaw) else moves.joinToString(" ")
            probes.add(parseProbe(label, moves, board))
        }
        if (probes.size >= 4) break
    }
    return probes.ifEmpty { probesByWords(text, board) }
}
