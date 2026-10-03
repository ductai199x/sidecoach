package com.sidecoach.profile

import com.sidecoach.analysis.later
import com.sidecoach.analysis.phaseOf
import com.sidecoach.analysis.terminalExpectedScore
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameReview
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.ProfileSnapshot
import com.sidecoach.events.Recommendation
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.Trigger
import com.sidecoach.events.UserUtterance
import com.sidecoach.vision.Pieces
import com.sidecoach.voice.Py
import java.math.BigDecimal
import java.math.MathContext
import java.util.UUID

/**
 * The profile stage: the events of a game in, the record, the evidence and the review out
 * (DESIGN sections 5.6 and 5.7). `sidecoach/profile/profiler.py`.
 *
 * It keeps its own account of each game from the same events the coach reads -- the moves,
 * the engine's verdicts, the analyses of the positions they lead to, the clocks -- and writes
 * everything to the store as it happens, each move with the facts a question might later
 * need. A mistake or a blunder is placed on the two axes (Causes) once its assessment and the
 * analysis of the position it left have both arrived, and becomes evidence. When the game
 * ends the review is built from the record (Review) and the snapshot of the whole profile is
 * refreshed, both as events for the coach and the panel -- once the engine has caught up with
 * the last moves, or after a grace period, or when the session closes, whichever comes first.
 */
class Profiler(
    val store: Store,
    sessionId: String? = null,
    ts: Double = 0.0,
    /** The puzzles to practise from, when there are any. */
    val puzzles: PuzzleStore? = null,
    /** The draw among the puzzles of a recommendation: injected so a test can fix it. */
    private val shuffle: (MutableList<Puzzle>) -> Unit = { it.shuffle() },
) {
    val sessionId: String = sessionId ?: UUID.randomUUID().toString().replace("-", "").take(8)

    private data class Key(val segment: Int, val ply: Int) : Comparable<Key> {
        override fun compareTo(other: Key): Int =
            compareValuesBy(this, other, { it.segment }, { it.ply })
    }

    private class Ply(val move: MovePlayed) {
        var moveId: Long = 0
        var assessment: MoveAssessment? = null
        var clockMs: Int? = null  // the mover's clock when the move was made
        var phase: String = "middlegame"
        var cause: String? = null
        var motif: String? = null
        var placed = false  // the evidence written
    }

    private class Game(
        val gameId: String,
        val userSide: Pieces.Side,
        val startFen: String,
        val startedAt: Double,
    ) {
        var segment = 0
        val plies = HashMap<Key, Ply>()
        /** The analysis of the position after (segment, ply); ply 0 is the segment's base. */
        val analyses = HashMap<Key, Analysis>()
        var opening: String? = null
        var lastFen: String = STARTING_FEN
        var timeControl: String? = null
        var ended = false
        /** The mistake the coach last gave a lesson on, and when. */
        var lesson: Pair<Key, Double>? = null
        /** The game is over; its review waits for the last verdicts. */
        var end: GameEnded? = null
        var provisional = false  // a position, not yet known to be a game
        /** Joined from the middle: its clocks say how much time was left, not the time control. */
        var joined = false
        var endedAt = 0.0
        var reviewed = false
        /** The game's phase so far: a phase reached is never left ([gamePhases]). */
        var phase = "opening"
    }

    private var game: Game? = null

    // The biggest cost (Stats.biggestCost) as of the games it was made from: read again only when
    // a game's facts change, not at every snapshot. `_cost`.
    private var cost: Pair<Pair<String?, List<Triple<String, String, String?>>>, org.json.JSONObject?>? = null
    private var clock: Pair<Int?, Int?> = null to null
    private val reviews = ArrayList<GameReview>()
    private var attempt: Attempt? = null
    private var attemptGame: String? = null
    private val recommendedFor = HashMap<String, String>()
    private var side = Pieces.Side.WHITE  // the player's side in the position last started

    init {
        store.startSession(this.sessionId, ts)
        store.purgeUnfinished(PROVISIONAL_PLIES)
    }

    // ---- events ------------------------------------------------------------------------------

    /** Consume one event; the events the profile emits in return, if any. */
    fun onEvent(event: Event, ts: Double): List<Event> {
        if (event is GameStarted) return flush(ts) + start(event, ts)
        if (event is ClockObservation) {
            // A box not read is no news of that side (`clock_observation`): its clock stays as
            // last read, not set to nothing (the review of 2026-10-01).
            clock = (event.whiteMs ?: clock.first) to (event.blackMs ?: clock.second)
            val game = game
            // A game joined from the middle took the time left for its control (the review of
            // 2026-10-01): it has none from its clocks.
            if (game != null && game.timeControl == null && !game.ended && !game.joined) {
                game.timeControl = timeControlWords(event.whiteMs, event.blackMs)
                if (game.timeControl != null) store.setGame(game.gameId, mapOf("time_control" to game.timeControl))
            }
            return emptyList()
        }
        val game = game
        if (game == null) {
            if (event is MovePlayed && event.gameId == attemptGame) return puzzleMove(event, ts)
            if (event is Resync && event.gameId == attemptGame) return anotherPosition(event, ts)
            return emptyList()
        }
        if (game.ended && (event is MovePlayed || event is Takeback || event is Resync || event is GameEnded)) {
            return emptyList()
        }
        if (removed(game)) return emptyList()
        if (event is Analysis && event.gameId != game.gameId) return emptyList()  // the engine's late word
        if (event is MoveAssessment && event.gameId != game.gameId) return emptyList()
        return when (event) {
            is MovePlayed -> move(game, event, ts)
            is Analysis -> {
                if (event.provisional) return emptyList()  // the record takes the full depth, which follows
                analysis(game, event) + maybeReview(game, ts)
            }
            is MoveAssessment -> assessment(game, event) + maybeReview(game, ts)
            is Takeback -> takeback(game, event)
            is Resync -> {
                if (game.provisional) return anotherPosition(event, ts)
                game.segment += 1
                game.lastFen = event.fen
                emptyList()
            }
            is CoachMessage -> {
                val ply = event.ply
                val at = if (ply != null) game.plies[Key(game.segment, ply)] else null
                store.addDialogue(game.gameId, ts, "coach", event.trigger.slug, event.text, ply, at?.moveId)
                if (event.trigger == Trigger.POST_MISTAKE && ply != null) {
                    // The lesson is about the move played from `ply`: the next ply's record.
                    game.lesson = Key(game.segment, ply + 1) to ts
                }
                emptyList()
            }
            is UserUtterance -> {
                store.addDialogue(game.gameId, ts, "user", event.source, event.text, null)
                selfReport(game, event.text, ts)
                emptyList()
            }
            is GameEnded -> end(game, event, ts)
            else -> emptyList()
        }
    }

    private fun start(event: GameStarted, ts: Double): List<Event> {
        dropIfProvisional()
        side = event.userSide
        // The clocks read in the game before are not this one's: an untimed game after a blitz
        // game finished on 4 seconds had every mistake blamed on the clock (the review of
        // 2026-10-01).
        clock = null to null
        if (event.kind != "game") {
            // A puzzle, an analysis board, or a game joined from the middle. A puzzle of the
            // store is followed for its outcome, not as a game; anything else is followed as
            // a game provisionally (see PROVISIONAL_PLIES).
            game = null
            attempt = null
            attemptGame = null
            if (event.kind != "position") return emptyList()
            if (puzzles != null) {
                attempt = puzzleAt(event.startFen)
                attemptGame = if (attempt != null) event.gameId else null
                if (attempt != null) return emptyList()
            }
        }
        game = Game(event.gameId, event.userSide, event.startFen, ts).apply {
            lastFen = event.startFen
            provisional = event.kind != "game"
            joined = event.kind != "game"
        }
        store.startGame(event.gameId, ts, event.userSide.name.lowercase(), event.startFen, sessionId)
        return emptyList()
    }

    /**
     * A position adopted over one that is not a game of the record: most often the next puzzle
     * of a run, which the site puts up with no start the reconciler could see, so it keeps the
     * game it had. It is followed as a position of its own, as a start would be: the plies of a
     * run of puzzles never add up to PROVISIONAL_PLIES (Tai's Pixel, 2026-09-24: after a few
     * puzzles a mate in three was reviewed as "won by checkmate in 3 moves ... over 5 games",
     * its misses kept as his habits), and a puzzle of the store is recognised on every one of
     * them, not only the first. A game joined late that needs a resync before it qualifies
     * starts its count again there.
     */
    private fun anotherPosition(event: Resync, ts: Double): List<Event> {
        val game = game
        if (game != null && game.plies.isEmpty() && game.startFen == event.fen) {
            return emptyList()  // the position just begun, announced again
        }
        return start(GameStarted(event.gameId, side, event.fen, kind = "position"), ts)
    }

    /** A game of the record that has ended and still waits for its review. */
    val gameOverUnreviewed: Boolean
        get() {
            val game = game
            return game != null && game.end != null && !game.reviewed
        }

    /**
     * Whether the current position counts as a game of the record: begun from the start, or a
     * joined position that has run PROVISIONAL_PLIES.
     */
    val followingAGame: Boolean
        get() {
            val game = game
            return game != null && !game.provisional
        }

    /** A provisional game that never qualified leaves nothing in the record. */
    private fun dropIfProvisional() {
        val game = game
        if (game != null && game.provisional) {
            store.deleteGame(game.gameId)
            this.game = null
        }
    }

    private fun move(game: Game, event: MovePlayed, ts: Double): List<Event> {
        val key = Key(game.segment, event.ply)
        val ply = Ply(event)
        val before = Board(event.fenBefore)
        val after = Board(event.fenAfter)
        val move = Move.fromUci(event.uci)
        val mover = before.turn
        ply.clockMs = if (mover == WHITE) clock.first else clock.second
        game.phase = later(game.phase, phaseOf(before))
        ply.phase = game.phase
        game.plies[key] = ply
        game.lastFen = event.fenAfter
        val user = game.userSide.color
        ply.moveId = store.addMove(
            game.gameId,
            game.segment,
            event.ply,
            moveNumber = before.fullmoveNumber,
            san = event.san,
            uci = event.uci,
            byUser = event.byUser,
            fenBefore = event.fenBefore,
            fenAfter = event.fenAfter,
            phase = ply.phase,
            thinkMs = event.thinkMs,
            clockMs = ply.clockMs,
            capture = if (before.isLegal(move)) before.isCapture(move) else false,
            givesCheck = after.isCheck(),
            material = materialBalance(after, user),
            ts = ts,
        )
        if (game.provisional && game.plies.size >= PROVISIONAL_PLIES) {
            game.provisional = false  // long enough to be no puzzle: a game, joined late
        }
        return emptyList()
    }

    /** The expected score of the analysed position from the player's side. */
    private fun scoreForUser(game: Game, event: Analysis): Double? {
        val board = Board(event.fen)
        val score = if (event.lines.isNotEmpty()) {
            event.lines[0].expectedScore
        } else {
            terminalExpectedScore(board) ?: return null
        }
        val user = game.userSide.color
        return if (board.turn == user) score else 1.0 - score
    }

    private fun analysis(game: Game, event: Analysis): List<Event> {
        val key = Key(game.segment, event.ply)
        game.analyses[key] = event
        if (!event.opening.isNullOrEmpty()) game.opening = event.opening
        val ply = game.plies[key]
        if (ply != null && ply.moveId != 0L) {
            store.analyseMove(
                ply.moveId,
                scoreAfter = scoreForUser(game, event),
                allowedTags = event.tags.map { it.name }.filter { it in Causes.TACTICAL },
            )
        }
        return place(game, key)
    }

    private fun assessment(game: Game, event: MoveAssessment): List<Event> {
        val key = Key(game.segment, event.ply)
        val ply = game.plies[key] ?: return emptyList()
        ply.assessment = event
        val before = Board(ply.move.fenBefore)
        val faced = game.analyses[Key(game.segment, event.ply - 1)]
        val available = faced?.tags?.map { it.name }?.filter { it in Causes.TACTICAL }?.toSortedSet()?.toList() ?: emptyList()
        val bestUci = event.best.firstOrNull()
        var bestSan: String? = null
        if (bestUci != null) {
            val bestMove = Move.fromUciOrNull(bestUci)
            if (bestMove != null && before.isLegal(bestMove)) bestSan = before.san(bestMove)
        }
        store.assessMove(
            ply.moveId,
            label = event.label?.name?.lowercase(),
            severity = event.severity.name.lowercase(),
            expectedLoss = event.expectedLoss,
            bestUci = bestUci,
            bestSan = bestSan,
            accuracy = event.accuracy,
            tags = event.tags.map { it.name },
            availableTags = available,
            depth = event.depth,
            phase = ply.phase,
        )
        if (!ply.move.byUser) return emptyList()
        return place(game, key)
    }

    /** Place a mistake on the two axes once its assessment and the analysis after it are in. */
    private fun place(game: Game, key: Key): List<Event> {
        val ply = game.plies[key] ?: return emptyList()
        val assessment = ply.assessment ?: return emptyList()
        if (ply.placed || !ply.move.byUser) return emptyList()
        if (assessment.severity.name.lowercase() !in Review.ERRORS) return emptyList()
        val after = game.analyses[key] ?: return emptyList()
        return listOf(evidence(game, key, ply, after))
    }

    private fun evidence(game: Game, key: Key, ply: Ply, after: Analysis): ProfileEvidence {
        val assessment = ply.assessment!!
        val (segment, number) = key
        val before = Board(ply.move.fenBefore)
        val faced = game.analyses[Key(segment, number - 1)]
        val previous = game.plies[Key(segment, number - 1)]
        val leftBook = previous != null &&
            previous.assessment != null &&
            previous.assessment!!.label == Label.BOOK &&
            assessment.label != Label.BOOK
        val reply = after.lines.firstOrNull()?.moves?.firstOrNull()
        val (cause, motif) = Causes.classify(
            before,
            Move.fromUci(ply.move.uci),
            playedTags = assessment.tags,
            availableTags = faced?.tags ?: emptyList(),
            allowedTags = after.tags,
            reply = reply,
            replyMate = after.lines.firstOrNull()?.mate,
            phase = ply.phase,
            leftBook = leftBook,
            clockMs = ply.clockMs,
            missedChance = assessment.label == Label.MISS,
        )
        ply.cause = cause
        ply.motif = motif
        ply.placed = true
        val loss = assessment.expectedLoss
        store.addEvidence(
            ply.moveId,
            gameId = game.gameId,
            segment = segment,
            ply = number,
            phase = ply.phase,
            expectedLoss = loss,
            cause = cause,
            motif = motif,
            timePressure = cause == Causes.TIME_PRESSURE,
        )
        return ProfileEvidence(
            gameId = game.gameId,
            ply = number,
            phase = ply.phase,
            expectedLoss = loss,
            motif = motif,
            cause = cause,
            timePressure = cause == Causes.TIME_PRESSURE,
        )
    }

    private fun takeback(game: Game, event: Takeback): List<Event> {
        val plies = game.plies.keys.filter { it.segment == game.segment }.sorted()
        if (plies.isEmpty()) return emptyList()
        val keep = maxOf(0, plies.size - event.plies)
        val fromPly = if (keep < plies.size) plies[keep].ply else plies.last().ply + 1
        for (key in plies.drop(keep)) {
            game.plies.remove(key)
            game.analyses.remove(key)
        }
        store.withdrawMoves(game.gameId, game.segment, fromPly)
        game.lastFen = event.fen
        return emptyList()
    }

    private fun end(game: Game, event: GameEnded, ts: Double): List<Event> {
        if (game.provisional) {
            dropIfProvisional()  // over before it could be a game: a puzzle
            return emptyList()
        }
        game.ended = true
        game.end = event
        game.endedAt = ts
        store.endGame(game.gameId, ts, event.result, event.reason, pgn = event.pgn, finalFen = event.finalFen, opening = game.opening)
        return maybeReview(game, ts)
    }

    /**
     * Whether the player removed the game from their record once it ended (the Games list offers
     * only finished games): nothing more is kept of it -- the engine's late verdicts, the
     * coach's words and the review were rows of a game that was gone, and the profile's cells
     * still counted them (the review of 2026-10-01). It is not reviewed either. `_removed`.
     */
    private fun removed(game: Game): Boolean {
        if (!game.ended || !store.isRemoved("${Store.WATCHED}${game.gameId}")) return false
        game.reviewed = true
        return true
    }

    /**
     * The review of a finished game, once every move of the player's has its verdict or the
     * grace period is up; [force] builds it from what is there.
     */
    private fun maybeReview(game: Game, ts: Double, force: Boolean = false): List<Event> {
        val end = game.end ?: return emptyList()
        if (game.reviewed || removed(game)) return emptyList()
        val waiting = game.plies.values.any { it.move.byUser && it.assessment == null }
        if (waiting && !force && ts - game.endedAt < REVIEW_GRACE_S) return emptyList()
        game.reviewed = true
        val reason = reasonWords(end)
        val review = review(game, end.result, reason)
        var final = mapOf("1-0" to 1.0, "0-1" to 0.0, "1/2-1/2" to 0.5)[end.result]
        if (final != null && game.userSide == Pieces.Side.BLACK) final = 1.0 - final
        store.endGame(
            game.gameId,
            game.endedAt,
            end.result,
            reason,
            pgn = end.pgn,
            finalFen = end.finalFen,
            accuracy = review.accuracy,
            finalScore = final,
            opening = game.opening,
        )
        reviews.add(review)
        return listOf(review) + recommend(game, review, ts) + snapshot()
    }

    // ---- puzzles (DESIGN section 5.7) ---------------------------------------------------------

    fun puzzleLevel(): Int = Puzzles.puzzleLevel(store)

    /**
     * A few puzzles for the motif that costs the player most: the profile's worst motif cell
     * once it has evidence enough, else this game's worst mistake's motif.
     */
    private fun recommend(game: Game, review: GameReview, ts: Double): List<Event> {
        val puzzles = puzzles ?: return emptyList()
        if (puzzles.count() == 0) return emptyList()
        val (motif, reason) = weakMotif(review)
        if (motif == null) return emptyList()
        val level = puzzleLevel()
        val recent = store.attempts(limit = 200)
        val solvedLately = recent.filter { (_, whenTs, solved) -> solved && ts - whenTs < RETRY_S }.map { it.first }.toSet()
        val failed = recent.filter { (_, whenTs, solved) -> !solved && ts - whenTs >= RETRY_FAIL_S }.map { it.first }
        val chosen = puzzles.pick(motif, level, exclude = solvedLately, first = failed, shuffle = shuffle)
        if (chosen.isEmpty()) return emptyList()
        for (puzzle in chosen) recommendedFor[puzzle.id] = motif
        val ids = chosen.map { it.id }
        store.addRecommendation(game.gameId, ts, motif, reason, level, ids)
        return listOf(
            Recommendation(
                gameId = game.gameId,
                motif = motif,
                motifWords = Insights.motifWords(motif),
                reason = reason,
                level = level,
                puzzles = chosen.map { it.ref() },
            ),
        )
    }

    /**
     * The motif to practise: the one missed most often over the games, as the insight line rates
     * it; else the game's turning point. The profile's cells keep the misses and the chances
     * apart, and asking one cell for both matched none: every recommendation was the turning
     * point's (the parity audit, 2026-09-29). `Profiler._weak_motif`.
     */
    private fun weakMotif(review: GameReview): Pair<String?, String> {
        val worst = Insights.motifRates(store.cells()).maxByOrNull { it.rate }
        if (worst != null) {
            return worst.motif to
                // How often it was found, never a tally of misses (Tai, 2026-10-02).
                "${Insights.motifWords(worst.motif)}: found ${worst.seen - worst.missed} of ${worst.seen} times over your games"
        }
        for (point in review.turningPoints) {
            if (!point.motif.isNullOrEmpty()) {
                return point.motif to "${Insights.motifWords(point.motif)}: the turning point of this game"
            }
        }
        return null to ""
    }

    /**
     * An outcome judged elsewhere (a puzzle solved on the puzzle page's board): kept beside the
     * motif, the level moved; the event for the coach and the panel.
     */
    fun recordPuzzle(puzzleId: String, motif: String?, solved: Boolean, moves: List<String>, ts: Double): List<Event> =
        listOf(Puzzles.recordAttempt(store, puzzleId, motif, solved, moves, ts))

    /** The puzzle whose solver's position this is, a recommended one first. */
    private fun puzzleAt(fen: String): Attempt? {
        val puzzles = puzzles ?: return null
        val key = try {
            Puzzles.positionKey(fen)
        } catch (_: Exception) {
            return null
        }
        val found = puzzles.atPosition(key)
        if (found.isEmpty()) return null
        val puzzle = found.sortedBy { it.id !in recommendedFor }.first()
        return Attempt(puzzle, recommendedFor[puzzle.id])
    }

    private fun puzzleMove(event: MovePlayed, ts: Double): List<Event> {
        val attempt = attempt ?: return emptyList()
        val outcome = attempt.move(event.uci, byUser = event.byUser) ?: return emptyList()
        this.attempt = null
        return recordPuzzle(attempt.puzzle.id, attempt.motif, outcome == "solved", attempt.played, ts)
    }

    /** Time passing: a finished game whose verdicts never came is reviewed as it is. */
    fun tick(ts: Double): List<Event> {
        val game = game ?: return emptyList()
        return maybeReview(game, ts)
    }

    /**
     * Review a finished game now, from what is there: the session is closing, or the next game
     * is starting. A game of the record that never ended is ended with the result unknown
     * first, so its moves are still reviewed and kept.
     */
    fun flush(ts: Double): List<Event> {
        val game = game ?: return emptyList()
        if (game.end == null && !game.ended) {
            if (game.provisional || game.plies.size < PROVISIONAL_PLIES) {
                // Too little to be a game of the record: a puzzle, or a game stopped after a
                // few moves.
                store.deleteGame(game.gameId)
                this.game = null
                return emptyList()
            }
            val ended = end(
                game,
                GameEnded(game.gameId, result = "*", reason = "the end was not seen", pgn = null, finalFen = game.lastFen),
                ts,
            )
            if (ended.isNotEmpty()) return ended  // every verdict was in: reviewed on the spot
        }
        return maybeReview(game, ts, force = true)
    }

    // ---- the review and the snapshot -----------------------------------------------------------

    private fun userMoves(game: Game): List<Review.UserMove> =
        game.plies.entries.sortedBy { it.key }.mapNotNull { (key, ply) ->
            val a = ply.assessment
            if (!ply.move.byUser || a == null) {
                null
            } else {
                Review.UserMove(
                    ply = key.ply,
                    segment = key.segment,
                    san = ply.move.san,
                    uci = ply.move.uci,
                    fenBefore = ply.move.fenBefore,
                    label = a.label?.name?.lowercase(),
                    severity = a.severity.name.lowercase(),
                    expectedLoss = a.expectedLoss,
                    bestUci = a.best.firstOrNull(),
                    accuracy = a.accuracy,
                    phase = ply.phase,
                    cause = ply.cause,
                    motif = ply.motif,
                )
            }
        }

    private fun review(game: Game, result: String, reason: String?): GameReview = Review.buildReview(
        gameId = game.gameId,
        userSide = game.userSide,
        result = result,
        reason = reason,
        opening = game.opening,
        moves = userMoves(game),
        fullmoves = Board(game.lastFen).fullmoveNumber,
    )

    /** What costs the player most, by the rule of Stats' "Work on this". `_biggest_cost`. */
    private fun biggestCost(): org.json.JSONObject? {
        val stamp = Stats.costStamp(store)
        val kept = cost
        if (kept != null && kept.first == stamp) return kept.second
        return Stats.biggestCost(store).also { cost = stamp to it }
    }

    fun snapshot(): ProfileSnapshot {
        val rows = store.cells()
        val games = store.gamesCount()
        val last = store.lastSession(before = sessionId)
        return ProfileSnapshot(
            games = games,
            cells = Insights.asCells(rows),
            insights = Insights.insights(rows, games, biggestCost()),
            lastSession = last?.summary,
        )
    }

    /** The user's word on why a mistake happened. Returns whether there was one. */
    fun correct(gameId: String, segment: Int, ply: Int, cause: String?): Boolean {
        store.correctEvidence(gameId, segment, ply, cause) ?: return false
        val game = game
        if (game != null && game.gameId == gameId) game.plies[Key(segment, ply)]?.cause = cause
        return true
    }

    /**
     * The player's first words within SELF_REPORT_S of the coach's lesson on a mistake are
     * kept beside that mistake; when they say plainly what happened, the cause follows their
     * word (Causes.fromWords).
     */
    private fun selfReport(game: Game, text: String, ts: Double) {
        val (key, whenTs) = game.lesson ?: return
        game.lesson = null
        if (ts - whenTs > SELF_REPORT_S) return
        val ply = game.plies[key] ?: return
        if (!ply.placed || ply.moveId == 0L) return
        store.selfReport(ply.moveId, text)
        val said = Causes.fromWords(text)
        if (said != null && said != ply.cause) {
            store.correctEvidence(game.gameId, key.segment, key.ply, said)
            ply.cause = said
        }
    }

    /** Forget the game in progress and this session's reviews: the profile was reset. */
    fun forget() {
        game = null
        reviews.clear()
        recommendedFor.clear()
    }

    // ---- the session -------------------------------------------------------------------------

    /**
     * Close the session with a summary of its games, for the next session's coach. A game
     * still waiting for its review is reviewed as it stands first.
     */
    fun endSession(ts: Double): String? {
        flush(ts)
        var summary: String? = null
        if (reviews.isNotEmpty()) {
            val outcomes = LinkedHashMap<String, Int>()
            for (r in reviews) {
                val key = if (r.result != "1-0" && r.result != "0-1" && r.result != "1/2-1/2") {
                    "unfinished"
                } else {
                    val won = (r.result == "1-0") == r.userSide.color
                    if (r.result == "1/2-1/2") "drew" else if (won) "won" else "lost"
                }
                outcomes[key] = (outcomes[key] ?: 0) + 1
            }
            val parts = outcomes.map { (k, n) -> "$n $k" }
            val accs = reviews.mapNotNull { it.accuracy }
            var text = "${reviews.size} game${if (reviews.size != 1) "s" else ""} (${parts.joinToString(", ")})"
            if (accs.isNotEmpty()) text += ", accuracy ${Insights.fixed(Py.sum(accs) / accs.size, 0)}"
            val costs = LinkedHashMap<String, Double>()
            for (r in reviews) for ((cause, loss) in r.causes) costs[cause] = (costs[cause] ?: 0.0) + loss
            if (costs.isNotEmpty()) {
                val (cause, loss) = costs.entries.maxByOrNull { it.value }!!.toPair()
                text += "; most ground went to $cause (${Insights.fixed(loss, 2)})"
            }
            val worst = reviews.flatMap { it.turningPoints }.maxByOrNull { it.expectedLoss }
            if (worst != null) {
                val better = if (!worst.bestSan.isNullOrEmpty()) ", ${worst.bestSan} was better" else ""
                text += "; the worst moment was ${worst.san}$better"
            }
            summary = "$text."
        }
        store.endSession(sessionId, ts, summary)
        return summary
    }

    companion object {
        /** A finished game waits this long for the engine's last verdicts. */
        const val REVIEW_GRACE_S = 90.0

        /**
         * A position begun from a board read cold is followed as a game provisionally and kept
         * once this many plies have been played on it: no puzzle runs that long.
         */
        const val PROVISIONAL_PLIES = 16

        /** The player's first words within this long after a lesson are about that mistake. */
        const val SELF_REPORT_S = 60.0

        /** A puzzle solved this recently is not recommended again; one failed this long ago first. */
        const val RETRY_S = 14 * 24 * 3600.0
        const val RETRY_FAIL_S = 3600.0

        private val VALUES = mapOf(PAWN to 1, KNIGHT to 3, BISHOP to 3, ROOK to 5, QUEEN to 9)

        /** Material from [side]'s view, in pawns: positive is up. */
        fun materialBalance(board: Board, side: Boolean): Int {
            var total = 0
            for (piece in board.pieceMap().values) {
                val value = VALUES[piece.pieceType] ?: 0
                total += if (piece.colour == side) value else -value
            }
            return total
        }

        // A clock's first reading this close below a whole minute is that minute ([timeControlWords];
        // Library.timeClassOf reads the records kept before the same way).
        const val START_SLACK_MS = 5_000

        /**
         * The time control from the clock's first reading as a game began: "10 min", "3 min",
         * "30 s". A reading up to [START_SLACK_MS] short of a whole minute is that minute: the
         * first reading comes a second or two after the clock started (599000 was "9.98333 min",
         * a ten-minute game counted as blitz: the review of 2026-10-01). `time_control_words`.
         */
        fun timeControlWords(whiteMs: Int?, blackMs: Int?): String? {
            val start = listOfNotNull(whiteMs, blackMs).maxOrNull() ?: return null
            if (start <= 0) return null
            val whole = (start + 59_999) / 60_000
            if (whole * 60_000 - start <= START_SLACK_MS) return "$whole min"
            if (start >= 60_000) return "${general(start / 60_000.0)} min"
            return "${start / 1000} s"
        }

        /** Python's `f"{x:g}"`: six significant digits, no trailing zeros. */
        private fun general(value: Double): String {
            val rounded = BigDecimal(value).round(MathContext(6)).stripTrailingZeros()
            return rounded.toPlainString()
        }

        /**
         * The reason as the review says it: the reconciler's "game over on board" is a mate, a
         * stalemate or dead material, which the final position tells.
         */
        fun reasonWords(event: GameEnded): String? {
            if (event.reason != "game over on board" || event.finalFen.isNullOrEmpty()) return event.reason
            val board = Board(event.finalFen)
            if (board.isCheckmate()) return "checkmate"
            if (board.isStalemate()) return "stalemate"
            if (board.isInsufficientMaterial()) return "insufficient material"
            return event.reason
        }
    }
}
