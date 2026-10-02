package com.sidecoach.profile

import com.sidecoach.profile.GameSummaries.round
import com.sidecoach.profile.StatFacts.ENDGAMES
import com.sidecoach.profile.StatFacts.KINDS
import com.sidecoach.profile.StatFacts.STRUCTURES
import com.sidecoach.profile.StatFacts.THEMES
import com.sidecoach.profile.StatFacts.standing
import org.json.JSONArray
import org.json.JSONObject

/**
 * The player's games added up (docs/stats-plan.md): the Stats window's numbers, from the facts
 * each review keeps ([StatFacts]), for the player's own moves in the games the filter takes.
 * Every figure carries how many it counts (`n`), so the page can grey a small sample; the
 * positions behind a figure come from [positions] with the same selector.
 * `sidecoach/profile/stats.py`.
 *
 * Beside the player's figures: a rating a category (the game rating's model fitted on that
 * category's moves alone, `categories.json`), and every figure the peers' -- the same figure for
 * real players of the player's rating (of their play, when no game records it), from the same
 * functions run over the judged Lichess sides (`peers.json`; both fitted by the desktop's
 * `tools/review/fit_stats.py`, and shipped beside this class as resources) -- with a grade for
 * how far ahead or behind the player's figure is, in rating points.
 *
 * The numbers are built as the JSON the page reads, in the desktop's order. Python's
 * `statistics.fmean` is [fmean]: `math.fsum` over the count, an exactly rounded sum.
 */
object Stats {
    val BAD = listOf("mistake", "miss", "blunder")

    // A move under this much clock is played under time pressure, by the time control.
    val PRESSURE_MS = mapOf("bullet" to 10_000, "blitz" to 20_000, "rapid" to 60_000)
    val GALLERIES = listOf("brilliant", "great", "miss", "blunder")

    // The moves each category is rated on, and its trend and figures drawn from.
    val CATEGORIES: Map<String, (StatMoveRow) -> Boolean> = linkedMapOf(
        "openings" to { m -> m.phase == "opening" },
        "tactics" to { m -> m.chance != null },
        "strategy" to { m -> m.chance == null && m.label != "book" },
        "endgames" to { m -> m.phase == "endgame" },
    )
    private const val CATEGORIES_PATH = "categories.json"
    private const val PEERS_PATH = "peers.json"

    // A grade: how far ahead of the peers the player's figure is, in rating points (the peers'
    // line read backwards), as the kinds of move say it; below the last, a blunder.
    val GRADES = listOf(
        400 to "brilliant",
        250 to "great",
        125 to "best",
        40 to "excellent",
        -40 to "good",
        -125 to "inaccuracy",
        -250 to "mistake",
    )

    // A figure of a few is pulled towards the peers' as if it had so many more at their value; a
    // figure the peers' line hardly follows (its fit's weighted r² below this) gets no grade.
    const val SHRINK = 10
    const val FOLLOWS = 0.5

    // A category's rating from this many of its moves: a tactic chance comes about twice a game,
    // and ten games' chances still leave the rating some 280 points out (tools/review/fit_stats.py).
    const val CATEGORY_MOVES = 20

    // The games the rating's trend pools at each game (the game and those before it).
    const val RATING_GAMES = 10

    /** `Filter`: [timeClass] bullet, blitz, rapid, daily, null for all; the last [days], null for all time; [opponents] all, people, bots. */
    data class Filter(val timeClass: String? = null, val days: Int? = null, val opponents: String = "all")

    private fun clock(): Double = System.currentTimeMillis() / 1000.0

    /** The games the filter takes, whose player's side is known, oldest first. `_games`. */
    private fun games(store: Store, f: Filter, now: Double): List<StatGameRow> {
        val since = if (f.days != null && f.days != 0) now - f.days.toLong() * 86400 else null
        val out = ArrayList<StatGameRow>()
        for (g in store.statGames()) {
            if (g.userSide != "white" && g.userSide != "black") continue
            if (!f.timeClass.isNullOrEmpty() && g.timeClass != f.timeClass) continue
            if (since != null && (g.playedAt ?: 0.0) < since) continue
            if ((f.opponents == "people" && g.bot) || (f.opponents == "bots" && !g.bot)) continue
            out.add(g)
        }
        return out
    }

    /** The player's points from the game: 1, a half, 0. `_score`. */
    private fun score(result: String?, side: String?): Double? {
        if (result == "1/2-1/2") return 0.5
        if (result == "1-0" || result == "0-1") return if ((result == "1-0") == (side == "white")) 1.0 else 0.0
        return null
    }

    /** A mean, with how many it is of. `_figure`. */
    private fun figure(values: List<Double>): JSONObject =
        JSONObject().put("value", if (values.isNotEmpty()) round(fmean(values), 1) else JSONObject.NULL).put("n", values.size)

    /** `_rate`. */
    private fun rate(hits: Int, total: Int): JSONObject = JSONObject()
        .put("value", if (total != 0) round(100.0 * hits / total, 1) else JSONObject.NULL)
        .put("hits", hits)
        .put("n", total)

    /** An opening's family: the name before its variation, without the ECO code. `family`. */
    fun family(opening: String?): String? {
        if (opening.isNullOrEmpty()) return null
        val name = opening.replaceFirst(ECO, "")
        return com.sidecoach.voice.Py.strip(name.split(":")[0])
    }

    private val ECO = Regex("^[A-E]\\d\\d\\s+")

    /** How a game ended, from chess.com's and Lichess's Termination header. `termination_kind`. */
    fun terminationKind(text: String?): String {
        val t = (text ?: "").lowercase(java.util.Locale.ROOT)
        for ((word, kind) in TERMINATIONS) {
            if (word in t) return kind
        }
        return "other"
    }

    private val TERMINATIONS = listOf(
        "checkmate" to "checkmate",
        "resign" to "resignation",
        "on time" to "time",
        "time forfeit" to "time",
        "timeout" to "time",
        "abandon" to "abandoned",
        "agreement" to "agreement",
        "repetition" to "repetition",
        "stalemate" to "stalemate",
        "insufficient" to "insufficient material",
        "50" to "fifty moves",
    )

    /**
     * Games (their `key`, `user_side`, `result`...) and their moves, both sides'; the player's
     * moves apart; the player's rating in each game that records one ([userElo], by key). From
     * the store ([of]), or from any rows shaped as its. `_Data`.
     */
    class Data(val games: List<StatGameRow>, moves: List<StatMoveRow>, val userElo: Map<String, Int?> = emptyMap()) {
        private val movesByGame = HashMap<String, MutableList<StatMoveRow>>()
        private val allByGame = HashMap<String, MutableList<StatMoveRow>>()
        val moves: List<StatMoveRow>

        init {
            val byKey = games.associateBy { it.key }
            for (m in moves) {
                allByGame.getOrPut(m.key) { ArrayList() }.add(m)
                if (m.side == byKey.getValue(m.key).userSide) movesByGame.getOrPut(m.key) { ArrayList() }.add(m)
            }
            this.moves = games.flatMap { mine(it.key) }
        }

        /** The player's moves in a game (`moves_by_game`). */
        fun mine(key: String): List<StatMoveRow> = movesByGame[key].orEmpty()

        /** Both sides' moves in a game (`all_by_game`). */
        fun all(key: String): List<StatMoveRow> = allByGame[key].orEmpty()

        companion object {
            /**
             * The filter's games in the store, each with the player's rating in it when the game
             * records one (the library's copy of its headers). `_Data.of`.
             */
            fun of(store: Store, f: Filter, now: Double = clock()): Data {
                val games = games(store, f, now)
                val elo = HashMap<String, Int?>()
                for (g in games) {
                    val row = store.libraryGame(g.key)
                    elo[g.key] = if (row == null) null else if (g.userSide == "white") row.whiteElo else row.blackElo
                }
                return Data(games, store.statMoves(games.map { it.key }), elo)
            }
        }
    }

    /**
     * Every section's numbers for the filter's games, the ratings, and the peers beside each
     * figure. [now] is the clock the last so many days count back from. `stats`.
     */
    fun stats(store: Store, f: Filter, now: Double = clock()): JSONObject {
        val d = Data.of(store, f, now)
        val out = JSONObject().put(
            "filter",
            JSONObject()
                .put("time_class", f.timeClass ?: JSONObject.NULL)
                .put("days", f.days ?: JSONObject.NULL)
                .put("opponents", f.opponents),
        )
        val sections = figures(d)
        for (key in sections.keys()) out.put(key, sections.get(key))
        out.put("trends", trends(d))
        out.put("shapes", shapes(store, d))
        val level = level(d)
        out.put("ratings", ratings(d, level))
        val peers = peersLevel(d, level)
        out.put("peers", JSONObject.NULL)
        if (peers != null) {
            compare(out, peers.first)
            out.put("peers", JSONObject().put("rating", Performance.forSite(peers.first, site(d))).put("from", peers.second))
        }
        out.put("work", work(out.getJSONObject("costs")))
        return out
    }

    /** The sections' figures for [d]: the player's, or the peers' (the same code for both). `figures`. */
    fun figures(d: Data): JSONObject = JSONObject()
        .put("overview", overview(d))
        .put("openings", openings(d))
        .put("tactics", tactics(d))
        .put("strategy", strategy(d))
        .put("endgames", endgames(d))
        .put("time", time(d))
        .put("results", results(d))
        .put("galleries", JSONObject().apply { for (k in GALLERIES) put(k, d.moves.count { it.label == k }) })
        .put("costs", costs(d))

    /** Every figure of a section (an object with its `value` and `n`), by its dotted path. `walk`. */
    fun walk(section: Any?, path: String = ""): List<Pair<String, JSONObject>> {
        val out = ArrayList<Pair<String, JSONObject>>()
        fun go(inner: Any?, at: String) {
            if (inner !is JSONObject) return
            if (inner.has("value") && inner.has("n")) {
                out.add(at to inner)
                return
            }
            for (key in inner.keys()) go(inner.get(key), if (at.isNotEmpty()) "$at.$key" else key)
        }
        go(section, path)
        return out
    }

    // ---- what the player's game costs them, and what to work on ----------------------------------

    // An area's puzzles: its motif as the puzzle store names it (Lichess's themes); none where no
    // puzzle drills it (a quiet position, a tactic of no named kind).
    val MOTIFS = mapOf(
        "missed.checkmate" to "mate",
        "missed.fork" to "fork",
        "missed.pin" to "pin",
        "missed.discovery" to "discoveredAttack",
        "missed.trapped piece" to "trappedPiece",
        "missed.removing the defender" to "capturingDefender",
        "missed.free piece" to "hangingPiece",
        "missed.sacrifice" to "sacrifice",
        "allowed.checkmate" to "mate",
        "allowed.fork" to "fork",
        "allowed.pin" to "pin",
        "allowed.discovery" to "discoveredAttack",
        "allowed.trapped piece" to "trappedPiece",
        "allowed.removing the defender" to "capturingDefender",
        "allowed.free piece" to "hangingPiece",
        "opening" to "opening",
        "endgame.pawn" to "pawnEndgame",
        "endgame.rook" to "rookEndgame",
        "endgame.queen" to "queenEndgame",
        "endgame.other" to "endgame",
    )
    const val WORK = 3 // the areas Work on this names
    const val WORK_GAMES = 3 // the games before it names any
    const val WORK_MOVES = 3 // the moves an area needs before it is named

    /**
     * The one area a move's cost counts in, when it cost anything worth a word: a tactic it
     * missed (by kind), else, for a mistake, miss or blunder, a tactic it handed over (by kind),
     * the endgame's (by kind), the opening's, a quiet position's. `area`.
     */
    fun area(m: StatMoveRow): String? {
        if (!m.chance.isNullOrEmpty() && !m.found) return "missed.${m.chance}"
        if (m.label !in BAD) return null
        if (!m.allowed.isNullOrEmpty()) return "allowed.${m.allowed}"
        if (m.phase == "endgame") return "endgame.${m.endgame?.takeIf { it.isNotEmpty() } ?: "other"}"
        return if (m.phase == "opening") "opening" else "quiet"
    }

    /**
     * Each area's cost: the expected score its moves lost, a game (`n`, the games), and how many
     * moves it counts (`moves`). The cost ranks the areas and meets the peers'; it is no count of
     * points: a position lost can swing back, and the losses add up past the games lost ("11.1
     * points" for one area when 9 were lost in all, the review of 2026-10-01). `_costs`.
     */
    private fun costs(d: Data): JSONObject {
        val lost = HashMap<String, Double>()
        val count = HashMap<String, Int>()
        for (m in d.moves) {
            val where = area(m) ?: continue
            lost[where] = (lost[where] ?: 0.0) + m.loss
            count[where] = (count[where] ?: 0) + 1
        }
        val games = d.games.size
        val names = KINDS.map { "missed.$it" } + KINDS.map { "allowed.$it" } + "opening" + ENDGAMES.map { "endgame.$it" } + "quiet"
        val out = JSONObject()
        for (name in names) {
            val figure = JSONObject()
                .put("value", if (games != 0) round((lost[name] ?: 0.0) / games, 4) else JSONObject.NULL)
                .put("n", games)
                .put("moves", count[name] ?: 0)
            val dot = name.indexOf('.')
            if (dot >= 0) {
                val head = name.substring(0, dot)
                val group = out.optJSONObject(head) ?: JSONObject().also { out.put(head, it) }
                group.put(name.substring(dot + 1), figure)
            } else {
                out.put(name, figure)
            }
        }
        return out
    }

    /**
     * The areas that cost the player most ([WORK] of them), dearest first: each with its cost a
     * game and its peers' (when known), the moves it counts, and its puzzles' motif; its positions
     * are [positions]' "area" selector's. Nothing over fewer than [WORK_GAMES] games, and no area
     * of fewer than [WORK_MOVES] moves: one game's blunder is no area to work on. `work`.
     */
    fun work(costs: JSONObject): JSONArray {
        val ranked = walk(costs)
            .filter { (_, figure) ->
                !figure.isNull("value") && figure.getDouble("value") != 0.0 &&
                    figure.getInt("n") >= WORK_GAMES && figure.optInt("moves") >= WORK_MOVES
            }
            .sortedBy { (_, figure) -> -figure.getDouble("value") }
        val out = JSONArray()
        for ((path, figure) in ranked.take(WORK)) {
            out.put(
                JSONObject()
                    .put("area", path)
                    .put("cost", figure.get("value"))
                    .put("games", figure.get("n"))
                    .put("moves", figure.get("moves"))
                    .put("peers", figure.opt("peers") ?: JSONObject.NULL)
                    .put("grade", figure.opt("grade") ?: JSONObject.NULL)
                    .put("motif", MOTIFS[path] ?: JSONObject.NULL),
            )
        }
        return out
    }

    // ---- the ratings and the peers -----------------------------------------------------------------

    /**
     * Moves as the rating model reads them: the plies before each (the store counts White's
     * first move as ply 1), the loss, the mover's expected score. `_rated`.
     */
    private fun rated(moves: List<StatMoveRow>): List<Performance.Played> = moves.map { Performance.Played(it.ply - 1, it.loss, it.score) }

    /** The scale the page speaks: Lichess's when every game is a Lichess game. `_site`. */
    private fun site(d: Data): String? = if (d.games.isNotEmpty() && d.games.all { it.source == "lichess" }) "lichess" else null

    /**
     * The player's rating over all their moves here (Lichess rapid), their level of play. Null
     * for too few moves. `_level`.
     */
    private fun level(d: Data): Double? {
        if (d.moves.size < Performance.MIN_MOVES) return null
        return Performance.estimate(rated(d.moves)).rating.toDouble()
    }

    /**
     * Who the player is compared with, on Lichess's scale, and why: players of their rating (the
     * mean of the ratings their games against people record, as chess.com's peers are), else of
     * their level of play ([level]). Against weaker players accuracy comes easier, so one who
     * plays above their rating is ahead of their rating's peers and, figure for figure, often of
     * their play's too: their own rating is the comparison that holds. `_peers_level`.
     */
    private fun peersLevel(d: Data, level: Double?): Pair<Double, String>? {
        val rated = d.games.mapNotNull { g ->
            val elo = d.userElo[g.key]
            if (elo == null || elo == 0 || g.bot) null
            else if (g.source == "lichess") elo.toDouble() else Performance.lichess(elo.toDouble())
        }
        if (rated.isNotEmpty()) return fmean(rated) to "rating"
        return if (level == null) null else level to "play"
    }

    /** A peers' fitted line: the figure at a rating is `a + b * rating`, fitted on ratings `lo` to `hi`, its weighted r². */
    data class PeerLine(val a: Double, val b: Double, val r2: Double, val lo: Double, val hi: Double)

    private fun resource(name: String): String =
        (Stats::class.java.getResourceAsStream(name) ?: throw IllegalStateException("no $name beside Stats"))
            .bufferedReader(Charsets.UTF_8).use { it.readText() }

    /** `_category_models`: each category's fitted weights, per context, per bucket, the logit's (1, x, x^2). */
    private val categoryModels: Map<String, Array<Array<DoubleArray>>> by lazy {
        val models = JSONObject(resource(CATEGORIES_PATH))
        models.keys().asSequence().associateWith { name ->
            val w = models.getJSONObject(name).getJSONArray("weights")
            Array(w.length()) { c ->
                val context = w.getJSONArray(c)
                Array(context.length()) { b ->
                    val bucket = context.getJSONArray(b)
                    DoubleArray(bucket.length()) { bucket.getDouble(it) }
                }
            }
        }
    }

    /** `_peer_lines`: each figure's peers' line, by the figure's dotted path. */
    private val peerLines: Map<String, PeerLine> by lazy {
        val figures = JSONObject(resource(PEERS_PATH)).getJSONObject("figures")
        figures.keys().asSequence().associateWith { path ->
            val line = figures.getJSONObject(path)
            PeerLine(line.getDouble("a"), line.getDouble("b"), line.getDouble("r2"), line.getDouble("lo"), line.getDouble("hi"))
        }
    }

    /**
     * The player's rating over all their moves, and each category's over its own moves, on the
     * page's scale; null where the moves are too few. `_ratings`.
     */
    private fun ratings(d: Data, level: Double?): JSONObject {
        val site = site(d)
        val overall = if (level == null) null else Performance.forSite(level, site)
        val out = JSONObject().put(
            "overall",
            if (overall == null) JSONObject.NULL else JSONObject().put("rating", overall).put("moves", d.moves.size),
        )
        for ((name, select) in CATEGORIES) {
            val moves = d.moves.filter(select)
            if (moves.size < CATEGORY_MOVES) {
                out.put(name, JSONObject.NULL)
                continue
            }
            val p = Performance.estimate(rated(moves), model = categoryModels.getValue(name))
            out.put(
                name,
                JSONObject()
                    .put("rating", Performance.forSite(p.rating.toDouble(), site))
                    .put("low", Performance.forSite(p.low.toDouble(), site))
                    .put("high", Performance.forSite(p.high.toDouble(), site))
                    .put("moves", p.moves),
            )
        }
        return out
    }

    /**
     * The peers' figure at [level] (Lichess rapid) by their fitted line, and the player's grade:
     * how many rating points ahead of it the figure is (read back along the line), a figure of a
     * few pulled towards the peers' first. No grade where the line hardly follows the rating.
     * `grade`.
     */
    fun grade(value: Double, n: Int, line: PeerLine, level: Double): Pair<Double, String?> {
        val rating = minOf(maxOf(level, line.lo), line.hi) // the sample's range, not beyond
        val peer = line.a + line.b * rating
        if (line.r2 < FOLLOWS || line.b == 0.0) return peer to null
        val shrunk = (n * value + SHRINK * peer) / (n + SHRINK)
        val ahead = (shrunk - peer) / line.b
        return peer to (GRADES.firstOrNull { (edge, _) -> ahead >= edge }?.second ?: "blunder")
    }

    /** Every figure with a peers' line gets the peers' value and the player's grade, in place. `compare`. */
    fun compare(out: JSONObject, level: Double) {
        val lines = peerLines
        for ((path, figure) in walk(out)) {
            val line = lines[path]
            if (line == null || figure.isNull("value")) continue
            val (peer, kind) = grade(figure.getDouble("value"), figure.getInt("n"), line, level)
            figure.put("peers", round(peer, if (path.startsWith("tactics.allowed")) 2 else 1))
            figure.put("grade", kind ?: JSONObject.NULL)
        }
    }

    /**
     * The rating the player played the game at ([Performance]), on the scale of the game's site.
     * `_game_rating`.
     */
    private fun gameRating(game: StatGameRow, moves: List<StatMoveRow>): Int? {
        if (moves.size < Performance.MIN_MOVES) return null
        val site = if (game.source == "lichess") "lichess" else null
        return Performance.forSite(Performance.estimate(rated(moves)).rating.toDouble(), site)
    }

    /**
     * Game by game, oldest first: the player's rating as of each game (`v`, their moves over it
     * and the games before it, [RATING_GAMES] in all, pooled; `game`, the game's own), and each
     * category's figure in the games that have one (the opening's accuracy, the tactics found,
     * the strategic accuracy, the endgame's accuracy). `_trends`.
     *
     * Pooled, not a mean of each game's estimate: one game's estimate is pulled to the middle by
     * the prior it needs, so their mean is too (on the held-out sides of tools/review/fit_stats.py,
     * 300 to 400 points for players over 2100 or under 1200); the moves of ten games pooled are not.
     */
    private fun trends(d: Data): JSONObject {
        fun mean(moves: List<StatMoveRow>): Double? =
            if (moves.isNotEmpty()) round(fmean(moves.map { it.accuracy ?: 0.0 }), 1) else null

        fun found(moves: List<StatMoveRow>): Double? {
            val chances = moves.filter { !it.chance.isNullOrEmpty() }
            return if (chances.isNotEmpty()) round(100.0 * chances.count { it.found } / chances.size, 1) else null
        }

        val pickers: Map<String, (StatGameRow, List<StatMoveRow>) -> Double?> = linkedMapOf(
            "openings" to { _, ms -> mean(ms.filter { it.phase == "opening" }) },
            "tactics" to { _, ms -> found(ms) },
            "strategy" to { _, ms -> mean(ms.filter(CATEGORIES.getValue("strategy"))) },
            "endgames" to { _, ms -> mean(ms.filter { it.phase == "endgame" }) },
        )
        val out = JSONObject().put("rating", JSONArray())
        for (name in pickers.keys) out.put(name, JSONArray())
        val site = site(d)
        var recent: List<List<StatMoveRow>> = emptyList()
        for (g in d.games) {
            val moves = d.mine(g.key)
            val own = gameRating(g, moves)
            if (own != null) {
                recent = recent.takeLast(RATING_GAMES - 1) + listOf(moves)
                val pooled = Performance.estimate(rated(recent.flatten())).rating
                out.getJSONArray("rating").put(
                    JSONObject()
                        .put("t", g.playedAt ?: JSONObject.NULL)
                        .put("key", g.key)
                        .put("v", Performance.forSite(pooled.toDouble(), site))
                        .put("game", own),
                )
            }
            for ((name, pick) in pickers) {
                val value = pick(g, moves) ?: continue
                out.getJSONArray(name).put(JSONObject().put("t", g.playedAt ?: JSONObject.NULL).put("key", g.key).put("v", value))
            }
        }
        return out
    }

    /** `_overview`. */
    private fun overview(d: Data): JSONObject {
        val scores = d.games.map { score(it.result, it.userSide) }
        val accuracies = d.games.filter { d.mine(it.key).isNotEmpty() }.map { g -> fmean(d.mine(g.key).map { it.accuracy ?: 0.0 }) }
        return JSONObject()
            .put("games", d.games.size)
            .put("wins", scores.count { it == 1.0 })
            .put("draws", scores.count { it == 0.5 })
            .put("losses", scores.count { it == 0.0 })
            .put("accuracy", figure(accuracies))
            .put("moves", d.moves.size)
    }

    /** Whether the player played the game's first move out of book. `_left_book_first`. */
    private fun leftBookFirst(moves: List<StatMoveRow>, side: String?): Boolean {
        val first = moves.firstOrNull { it.label != "book" }
        return first != null && first.side == side
    }

    /** `_openings`. */
    private fun openings(d: Data): JSONObject {
        val openingMoves = d.moves.filter { it.phase == "opening" }
        val mistakes = d.games.map { g -> d.mine(g.key).count { it.phase == "opening" && it.label in BAD }.toDouble() }
        val book = d.games.filter { leftBookFirst(d.all(it.key), it.userSide) }
            .map { g -> d.mine(g.key).count { it.label == "book" }.toDouble() }
        val groups = LinkedHashMap<Pair<String, String?>, MutableList<StatGameRow>>()
        for (g in d.games) {
            val name = family(g.opening)
            if (!name.isNullOrEmpty()) groups.getOrPut(name to g.userSide) { ArrayList() }.add(g)
        }
        val byOpening = ArrayList<JSONObject>()
        for ((group, games) in groups) {
            val (name, side) = group
            val scores = games.mapNotNull { score(it.result, side) }
            val accuracies = games.filter { d.mine(it.key).isNotEmpty() }.map { g -> fmean(d.mine(g.key).map { it.accuracy ?: 0.0 }) }
            byOpening.add(
                JSONObject()
                    .put("opening", name)
                    .put("side", side ?: JSONObject.NULL)
                    .put("games", games.size)
                    .put("score", if (scores.isNotEmpty()) round(100 * fmean(scores), 1) else JSONObject.NULL)
                    .put("accuracy", if (accuracies.isNotEmpty()) round(fmean(accuracies), 1) else JSONObject.NULL),
            )
        }
        val sorted = byOpening.sortedWith(compareBy({ -it.getInt("games") }, { it.getString("opening") }))
        return JSONObject()
            .put("accuracy", figure(openingMoves.map { it.accuracy ?: 0.0 }))
            .put("mistakes", figure(mistakes))
            .put("book_moves", figure(book))
            .put("by_opening", JSONArray(sorted))
    }

    /** `_tactics`. */
    private fun tactics(d: Data): JSONObject {
        val found = JSONObject()
        for (kind in KINDS) {
            found.put(kind, rate(d.moves.count { it.chance == kind && it.found }, d.moves.count { it.chance == kind }))
        }
        val total = d.moves.size
        val allowed = JSONObject()
        for (kind in KINDS) {
            val count = d.moves.count { it.allowed == kind }
            allowed.put(
                kind,
                JSONObject()
                    .put("value", if (total != 0) round(1000.0 * count / total, 2) else JSONObject.NULL)
                    .put("hits", count)
                    .put("n", total),
            )
        }
        val chances = d.moves.filter { !it.chance.isNullOrEmpty() }
        return JSONObject()
            .put("found", found)
            .put("allowed", allowed)
            .put("overall", rate(chances.count { it.found }, chances.size))
    }

    /** `_strategy`. */
    private fun strategy(d: Data): JSONObject {
        val quiet = d.moves.filter(CATEGORIES.getValue("strategy"))
        val out = JSONObject()
        for (structure in listOf("all") + STRUCTURES) {
            val moves = if (structure == "all") quiet else quiet.filter { it.structure == structure }
            val row = JSONObject()
                .put("overall", figure(moves.map { it.accuracy ?: 0.0 }))
                .put("attacking", figure(moves.filter { standing(it.score) == "better" }.map { it.accuracy ?: 0.0 }))
                .put("defending", figure(moves.filter { standing(it.score) == "worse" }.map { it.accuracy ?: 0.0 }))
            for (theme in THEMES) {
                row.put(theme, figure(moves.filter { theme in it.themes.split(",") }.map { it.accuracy ?: 0.0 }))
            }
            out.put(structure, row)
        }
        return out
    }

    /** `_endgames`. */
    private fun endgames(d: Data): JSONObject {
        val entered = linkedMapOf<String, MutableList<Double>>("better" to ArrayList(), "equal" to ArrayList(), "worse" to ArrayList())
        for (g in d.games) {
            val first = d.all(g.key).firstOrNull { it.phase == "endgame" }
            val score = score(g.result, g.userSide)
            if (first == null || score == null) continue
            val mine = if (first.side == g.userSide) first.score else 1.0 - first.score
            entered.getValue(standing(mine)).add(score)
        }
        val endgame = d.moves.filter { it.phase == "endgame" }
        val win = JSONObject()
        for ((k, v) in entered) {
            win.put(k, JSONObject().put("value", if (v.isNotEmpty()) round(100 * fmean(v), 1) else JSONObject.NULL).put("n", v.size))
        }
        val accuracy = JSONObject().put("overall", figure(endgame.map { it.accuracy ?: 0.0 }))
        for (kind in ENDGAMES) accuracy.put(kind, figure(endgame.filter { it.endgame == kind }.map { it.accuracy ?: 0.0 }))
        return JSONObject().put("win", win).put("accuracy", accuracy)
    }

    /** `_time`. */
    private fun time(d: Data): JSONObject {
        val timed = d.moves.filter { it.spentMs != null }
        val byPhase = JSONObject()
        for (phase in listOf("opening", "middlegame", "endgame")) {
            byPhase.put(phase, figure(timed.filter { it.phase == phase }.map { it.spentMs!! / 1000.0 }))
        }
        val pressed = ArrayList<StatMoveRow>()
        val calm = ArrayList<StatMoveRow>()
        for (g in d.games) {
            val limit = PRESSURE_MS[g.timeClass ?: ""] ?: continue
            for (m in d.mine(g.key)) {
                val clock = m.clockMs ?: continue
                (if (clock < limit) pressed else calm).add(m)
            }
        }
        var lostOnTime = 0
        var lostOnTimeWinning = 0
        for (g in d.games) {
            val score = score(g.result, g.userSide)
            // `_score(...)` read as a truth: a game won or drawn is no loss; one of no result is.
            if (terminationKind(g.termination) != "time" || (score != null && score != 0.0)) continue
            lostOnTime += 1
            val mine = d.mine(g.key)
            if (mine.isNotEmpty() && standing(mine.last().score) == "better") lostOnTimeWinning += 1
        }
        return JSONObject()
            .put("by_phase", byPhase)
            .put("pressure", rate(pressed.count { it.label in BAD }, pressed.size))
            .put("calm", rate(calm.count { it.label in BAD }, calm.size))
            .put("lost_on_time", lostOnTime)
            .put("lost_on_time_winning", lostOnTimeWinning)
    }

    /**
     * How the player's games went by their shape (`GameSummaries.shapeOf`, as each review's
     * summary has it: smooth, balanced, sudden...): how many, and won, drawn and lost -- the
     * plan's G, which kinds of game suit the player. Only the shapes their games had, in
     * SHAPES' order. `_shapes`.
     */
    private fun shapes(store: Store, d: Data): JSONArray {
        val counts = HashMap<String, IntArray>() // games, won, drawn, lost
        for (g in d.games) {
            val summary = store.review(g.key, g.strength)?.summary ?: continue
            val shape = runCatching { JSONObject(summary).let { if (it.isNull("shape")) null else it.optString("shape") } }.getOrNull()
            if (shape.isNullOrEmpty()) continue
            val c = counts.getOrPut(shape) { IntArray(4) }
            c[0] += 1
            val score = score(g.result, g.userSide)
            if (score != null) c[if (score == 1.0) 1 else if (score == 0.5) 2 else 3] += 1
        }
        return JSONArray().apply {
            for (s in GameSummaries.SHAPES) {
                val c = counts[s] ?: continue
                put(JSONObject().put("shape", s).put("games", c[0]).put("won", c[1]).put("drawn", c[2]).put("lost", c[3]))
            }
        }
    }

    /** `_results`. */
    private fun results(d: Data): JSONObject {
        val out = LinkedHashMap<String, IntArray>()
        for (g in d.games) {
            val score = score(g.result, g.userSide) ?: continue
            val outcome = if (score == 1.0) 0 else if (score == 0.5) 1 else 2
            out.getOrPut(terminationKind(g.termination)) { IntArray(3) }[outcome] += 1
        }
        return JSONObject().apply {
            for ((kind, counts) in out) put(kind, JSONObject().put("won", counts[0]).put("drawn", counts[1]).put("lost", counts[2]))
        }
    }

    // ---- the positions behind a figure --------------------------------------------------------------

    private val SELECT: Map<String, (StatMoveRow, String) -> Boolean> = mapOf(
        "opening.mistakes" to { m, _ -> m.phase == "opening" && m.label in BAD },
        "tactics.found" to { m, kind -> m.chance == kind && m.found },
        "tactics.missed" to { m, kind -> m.chance == kind && !m.found },
        "tactics.allowed" to { m, kind -> m.allowed == kind },
        "strategy.mistakes" to { m, structure ->
            m.chance == null && m.label in BAD && (structure == "" || structure == "all" || m.structure == structure)
        },
        "endgame.mistakes" to { m, kind ->
            m.phase == "endgame" && m.label in BAD && (kind == "" || kind == "overall" || m.endgame == kind)
        },
        "time.pressure" to { m, _ -> m.label in BAD }, // of the moves under pressure
        "gallery" to { m, label -> m.label == label },
        "area" to { m, name -> area(m) == name },
    )
    const val POSITIONS_MAX = 60

    /**
     * The player's moves behind a figure, the latest first: [what] a selector of `_SELECT`, [arg]
     * its kind (a tactic's, an endgame's, a structure, a gallery's label). `positions`.
     */
    fun positions(store: Store, f: Filter, what: String, arg: String = "", now: Double = clock()): JSONArray {
        val select = SELECT[what] ?: return JSONArray()
        val d = Data.of(store, f, now)
        val out = ArrayList<JSONObject>()
        for (g in d.games.asReversed()) {
            val limit = PRESSURE_MS[g.timeClass ?: ""]
            for (m in d.mine(g.key)) {
                if (what == "time.pressure" && (limit == null || m.clockMs == null || m.clockMs >= limit)) continue
                if (select(m, arg)) {
                    out.add(
                        JSONObject()
                            .put("key", g.key)
                            .put("ply", m.ply)
                            .put("fen", m.fen)
                            .put("uci", m.uci)
                            .put("san", m.san)
                            .put("label", m.label ?: JSONObject.NULL)
                            .put("side", m.side)
                            .put("played_at", g.playedAt ?: JSONObject.NULL)
                            .put("opening", g.opening ?: JSONObject.NULL),
                    )
                }
            }
            if (out.size >= POSITIONS_MAX) break
        }
        return JSONArray(out.take(POSITIONS_MAX))
    }

    // ---- Python's float means ----------------------------------------------------------------------

    /** `statistics.fmean`: `math.fsum` over the count. */
    fun fmean(values: List<Double>): Double = fsum(values) / values.size

    /**
     * `math.fsum`, as CPython computes it: Shewchuk's exact partials, then the top ones added
     * until the sum turns inexact, with a half-way case rounded to even across the partials.
     * Finite values only (the facts are).
     */
    fun fsum(values: Iterable<Double>): Double {
        val p = ArrayList<Double>()
        for (value in values) {
            var x = value
            var i = 0
            for (j in p.indices) {
                var y = p[j]
                if (Math.abs(x) < Math.abs(y)) {
                    val t = x
                    x = y
                    y = t
                }
                val hi = x + y
                val yr = hi - x
                val lo = y - yr
                if (lo != 0.0) p[i++] = lo
                x = hi
            }
            while (p.size > i) p.removeAt(p.size - 1) // ps[i:] = [x]
            if (x != 0.0) p.add(x)
        }
        var n = p.size
        var hi = 0.0
        if (n > 0) {
            var lo = 0.0
            hi = p[--n]
            while (n > 0) {
                val x = hi
                val y = p[--n]
                hi = x + y
                val yr = hi - x
                lo = y - yr
                if (lo != 0.0) break
            }
            if (n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0))) {
                val y = lo * 2.0
                val x = hi + y
                val yr = x - hi
                if (y == yr) hi = x
            }
        }
        return hi
    }
}
