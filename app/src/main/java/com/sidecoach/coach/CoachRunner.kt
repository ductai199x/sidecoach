package com.sidecoach.coach

import android.util.Log
import com.sidecoach.analysis.Computed
import com.sidecoach.analysis.ComputedFirst
import com.sidecoach.analysis.Engine
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.Event
import com.sidecoach.llm.KeyStore
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.LLMSettings
import com.sidecoach.voice.Py
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.makeProvider
import com.sidecoach.profile.Ask
import com.sidecoach.profile.Store
import com.sidecoach.ui.PanelEngine
import com.sidecoach.ui.PanelVoice
import org.json.JSONArray
import org.json.JSONObject
import com.sidecoach.ui.SettingsStore

/**
 * The coach alongside a game: the events in, the messages out. `cli._Coaching`.
 *
 * A model that cannot be made (no key yet, a preset unknown) or reached (the local server
 * down) leaves the coach on, offline: the board and the templates carry it until the
 * settings say otherwise (Tai, 2026-09-21: the app works without a model). With no model
 * there is nothing that would call one -- no claim extraction from the player's words, no
 * verifier -- and the templates and the board carry the coach.
 *
 * The coach has its own engine, as the desktop's: for the moves the player names, and for
 * the model's questions to the engine while it answers (`EngineTools`) -- without one the
 * phone's model could ask the engine nothing, and "ok and after that?" past a line's end
 * had no engine's word to go on (the parity audit, 2026-09-29). [engine] is built by the
 * caller (`Assessing.coachEngine`) and asked only what the analyser has not computed
 * ([computed], `cli._named_engine`): a second process only once a question needs a position
 * nothing reaches.
 */
class CoachRunner(
    settings: SettingsStore,
    enabled: Boolean = true,
    engine: Engine? = null,
    computed: Computed? = null,
) {

    private val coachEngine: ComputedFirst? = engine?.let { ComputedFirst(it, computed ?: Computed()) }

    val coach: Coach?
    var detail: String = "off"
        private set

    init {
        coach = if (enabled) build(settings) else null
    }

    /**
     * The coach itself, when it is wanted: `_Coaching.__init__` past its `if not enabled`.
     * The desktop builds it only when there is an engine to measure moves with.
     */
    private fun build(settings: SettingsStore): Coach? {
        val (model, detail) = provision(settings)
        val offline = model is OfflineProvider
        this.detail = "${model.name} ($detail)"
        return try {
            Coach(
                model,
                policy = InterjectionPolicy(settings.policy()),
                draftCheck = { text, board, after, us ->
                    checkDraft(text, board, after, us).map { "${it.text}: ${it.reason}" }
                },
                // The claims in their words are parsed at the routine effort: at medium every ask with
                // a claim waited on a second full model call first (the runtime audit, 2026-09-25).
                claims = if (offline) null else claimsCheck(routineProvider(settings, model)),
                verifier = if (offline) null else verifier(settings, model),
                engine = coachEngine?.let { EngineLike(it::analyse) },
                summariseEvery = settings.summariseEvery(),
                summariser = model,
                compactRecord = settings.compactRecord(),
                routineProvider = routineProvider(settings, model),
                verbalizedSampling = settings.verbalizedSampling(),
                localVerdicts = settings.localVerdicts(),
                lessonWaitS = LESSON_WAIT_S,
                persona = settings.persona(),
                rating = settings.rating(),
            ).also { it.start() }
        } catch (exc: Exception) {
            Log.w(TAG, "the coach did not start: $exc")
            null
        }
    }

    /**
     * The model changed in the settings: the providers are rebuilt and handed to the
     * running coach, with the same fallback to offline as at start. `_Coaching.reprovision`.
     */
    fun reprovision(settings: SettingsStore) {
        val coach = coach ?: return
        val (provider, detail) = provision(settings)
        val offline = provider is OfflineProvider
        coach.setProviders(
            provider,
            routineProvider = routineProvider(settings, provider),
            verifier = if (offline) null else verifier(settings, provider),
            claims = if (offline) null else claimsCheck(routineProvider(settings, provider)),
        )
        this.detail = "${provider.name} ($detail)"
    }

    /**
     * The model the settings name, and a line on it; the offline coach and the reason when
     * it cannot be made or reached. What `_Coaching.__init__` and `reprovision` share.
     */
    private fun provision(settings: SettingsStore): Pair<LLMProvider, String> {
        var why = ""
        var provider: LLMProvider? = null
        var detail = ""
        try {
            provider = makeProvider(
                settings.llm(),
                keys = KeyStore { name -> settings.secret(name) },
            )
        } catch (exc: Exception) {
            why = "no model: ${exc.message}"
        }
        if (provider != null) {
            val (ok, said) = try {
                provider.healthcheck()
            } catch (exc: Exception) {
                false to "$exc"
            }
            detail = said
            if (!ok) why = "model not reachable: $said"
        }
        if (why.isNotEmpty()) {
            provider = OfflineProvider()
            detail = provider.healthcheck().second + "; " + why
        }
        return provider!! to detail
    }

    val on: Boolean get() = coach != null

    fun events(events: List<Event>, ts: Double) {
        val coach = coach ?: return
        for (event in events) {
            try {
                coach.onEvent(event, ts)
            } catch (exc: Exception) {
                Log.w(TAG, "coach error on ${event::class.java.simpleName}: $exc")
                PanelEngine.emit("sys", "warn", "coach error: ${PanelEngine.pyRepr(exc)}")
            }
        }
    }

    fun tick(ts: Double) {
        coach?.tick(ts)
    }

    fun ask(text: String, ts: Double) {
        coach?.ask(text, ts)
    }

    private var drained = 0
    private var profileSeen = 0

    /**
     * The record the coach's model may query for a question about the player over their games
     * (`profile.ask`), for the session: its own model writes the SQL, read-only.
     */
    fun useRecord(store: Store?) {
        val coach = coach ?: return
        coach.profileAsk = if (store == null) {
            null
        } else {
            { question -> Ask.answer(question, store, coach.speaker.provider) }
        }
    }

    /** The coach's replies in flight, waited for at a Stop: `_Coaching.wait`. */
    fun wait(timeoutS: Double = 30.0) {
        val coach = coach ?: return
        val deadline = System.nanoTime() + (timeoutS * 1e9).toLong()
        while (coach.speaker.pending != 0 && System.nanoTime() < deadline) {
            drain()
            Thread.sleep(50)
        }
        drain()
    }

    /**
     * Where the coach's messages go when set (a puzzle room's log): `_Coaching.sink`. Left
     * null, they go onto the panel.
     */
    @Volatile var sink: ((CoachMessage) -> Unit)? = null

    private var failuresSeen = 0
    private var silence: String? = null

    /**
     * Whatever the coach has finished saying, onto the panel, with what the checks found:
     * `_Coaching.drain` then `panel._coach_meta`. Every turn of the conversation at debug (the
     * Conversation tab is built from them), the record queries a question ran, each message
     * with why it spoke and what it was checked against -- spoken by the voice's rules
     * (`PanelVoice.speak`) -- each model reply's figures, and any new model failure.
     */
    fun drain() {
        val coach = coach ?: return
        val sink = sink
        if (sink != null) {
            for (message in coach.poll()) sink(message)
            return
        }
        coach.poll() // records each reply as a turn of the conversation
        // Every message of the game's conversation, as the model will see it, at debug.
        val turns = synchronized(coach.lock) {
            coach.conversation.drainNew().filter { it.kind != "coach" }.map { entry ->
                val text = if (entry.kind == "ply" && entry.record != null) {
                    coach.conversation.renderPly(entry.record)
                } else {
                    entry.text
                }
                Triple(entry, text, entry.role)
            }
        }
        for ((entry, text, role) in turns) {
            PanelEngine.emit(
                "coach",
                "debug",
                "turn (${entry.kind}): ${PanelEngine.elide(text, 80)}",
                JSONObject().put("role", role).put("kind", entry.kind).put("body", text.take(6000)),
            )
        }
        // What the coach looked up in the record for a question about the player.
        val answers = coach.profileAnswers
        while (profileSeen < answers.size) {
            val (kind, answer) = answers[profileSeen++]
            if (kind == "error") {
                PanelEngine.emit("coach", "error", "record lookup failed: $answer")
                continue
            }
            answer as ProfileAnswer
            for ((i, exchange) in answer.exchanges.withIndex()) {
                exchange as Ask.Exchange
                val head = exchange.error ?: "${exchange.rows.size} rows"
                val sql = exchange.sql.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
                PanelEngine.emit(
                    "coach",
                    if (exchange.error != null) "warn" else "info",
                    "record query ${i + 1}: ${PanelEngine.elide(sql, 90)} — $head",
                    JSONObject().apply {
                        put("sql", exchange.sql)
                        put("columns", JSONArray(exchange.columns))
                        put("rows", JSONArray(exchange.rows.take(20).map { row -> JSONArray(row.map { it.toString() }) }))
                        put("error", exchange.error ?: JSONObject.NULL)
                    },
                )
            }
            if (answer.failed) PanelEngine.emit("coach", "warn", "record: the model wrote no answer; the rows stand in")
        }
        val spoken = coach.spoken
        while (drained < spoken.size) {
            val one = spoken[drained++]
            val message = one.message
            // Written down in the profile as the conversation (`panel._record_dialogue`).
            if (message.text.isNotEmpty()) PanelEngine.recordDialogue(message)
            val meta = arrayListOf(
                message.trigger.slug,
                "L${one.level.value}",
                if (message.ply != null) "ply ${message.ply}" else "no ply",
            )
            if (one.attempts != 0) meta.add("${one.attempts} attempt${if (one.attempts != 1) "s" else ""}")
            if (one.leaks.isNotEmpty()) {
                // For debugging: what the reply gave away of the answer, spoken anyway.
                meta.add("leak noted: " + one.leaks.joinToString("; "))
            }
            if (one.notes.isNotEmpty()) {
                // The same for its form: what it did against the form asked of it.
                meta.add("form noted: " + one.notes.joinToString("; "))
            }
            if (message.text.isEmpty()) {
                // The model's reply failed the checks and the quick answer already stands.
                PanelEngine.emit(
                    "coach",
                    "warn",
                    meta.joinToString(" · ") + " — model reply rejected; the quick answer stands",
                    JSONObject().apply {
                        put("trigger", message.trigger.slug)
                        put("disclosure", one.level.value)
                        put("ply", message.ply ?: JSONObject.NULL)
                        put("attempts", one.attempts)
                        put("problems", JSONArray(one.problems))
                    },
                )
            } else {
                if (one.problems.isNotEmpty() && !one.streamed) {
                    meta.add("draft rejected → template")
                } else if (one.problems.isNotEmpty()) {
                    meta.add("cut short: " + one.problems.joinToString("; "))
                }
                val action = if (one.streamed) "streamed" else PanelVoice.speak(message.trigger.slug, message.text, message.ply)
                when (action) {
                    "streamed" -> meta.add("streamed")
                    "stale" -> meta.add("not spoken: board moved on")
                    "skip" -> meta.add("not spoken: voice busy")
                    "text" -> meta.add("text: fast game")
                    "supersede" -> meta.add("cut in")
                }
                PanelEngine.said(message, one, meta.joinToString(" · "), action)
            }
            for ((index, call) in one.calls.withIndex()) {
                // What the model actually wrote, so a stump or a leak can be traced to its
                // source; a reply the token budget cut short is worth a warning.
                val finish = (call.stats["finish_reason"] as? String)?.ifEmpty { null } ?: "?"
                val tokens = call.stats["completion_tokens"]
                val seconds = call.stats["elapsed_s"] as? Number
                val first = call.stats["first_token_s"] as? Number
                if (first != null) PanelVoice.noteFirstToken(first.toDouble())
                @Suppress("UNCHECKED_CAST")
                val streamProblems = call.stats["stream_problems"] as? List<String>
                val summary = "model reply ${index + 1}: ${call.candidates} candidates, ${call.passed} passed" +
                    (if (tokens != null) ", $tokens tokens" else "") +
                    (if (seconds != null) " in ${fixed1(seconds.toDouble())}s" else "") +
                    (if (first != null) " (first token ${fixed1(first.toDouble())}s)" else "") +
                    (if (finish == "length") ", cut off by the token budget" else "") +
                    (if (!streamProblems.isNullOrEmpty()) "; stream stopped: ${streamProblems.joinToString("; ")}" else "")
                PanelEngine.emit(
                    "coach",
                    if (finish == "length") "warn" else "debug",
                    summary,
                    JSONObject().apply {
                        put("raw", PanelEngine.elide(call.raw, 2000))
                        put("prompt", call.prompt.takeLast(6000))
                        put("finish_reason", finish)
                        put("completion_tokens", tokens ?: JSONObject.NULL)
                        put("elapsed_s", seconds ?: JSONObject.NULL)
                        put("candidates", call.candidates)
                        put("passed", call.passed)
                    },
                )
            }
        }
        val speaker = coach.speaker
        if (speaker.failures > failuresSeen) {
            PanelEngine.emit("coach", "error", "LLM call failed: ${speaker.lastError}")
            failuresSeen = speaker.failures
        }
        // At debug, why the coach is quiet -- only when the reason changes. `_coach_silence`.
        val reason = coach.policy.lastReason
        if (reason != null && reason != silence) PanelEngine.emit("coach", "debug", "quiet: $reason")
        silence = reason
    }

    private fun fixed1(value: Double): String =
        java.math.BigDecimal(value).setScale(1, java.math.RoundingMode.HALF_EVEN).toPlainString()

    val pending: Int get() = coach?.speaker?.pending ?: 0

    fun close() {
        coach?.close()
        coachEngine?.close()
    }

    /** The user's stated reasons, checked against the board. `cli._claims_check`. */
    private fun claimsCheck(provider: LLMProvider): ClaimsCheck = ClaimsCheck { text, board, facts ->
        // Reasoning off: a local model spends the whole budget thinking otherwise, and the
        // claims come out the same in a fifth of the time.
        val found = extractClaims(text, board, provider, thinking = false)
        // What the board cannot decide tells the model nothing: "whose bishop? mine?" came to
        // it as "unclear, I cannot check that against the board; what do you expect to happen,
        // concretely?" (Tai's puzzle, 2026-09-28).
        verifyClaims(found, board, facts = facts)
            .filter { it.result != ClaimResult.UNCLEAR }
            .map { claimLine(it) }
    }

    /**
     * The verifier the settings ask for: none, the coach's own model, or another provider --
     * that preset with the model `coach.verify_model` names, and nothing of the coach's own
     * model settings (its model, its address, its effort). `cli._verifier`.
     */
    internal fun verifier(settings: SettingsStore, provider: LLMProvider): LLMProvider? {
        val kind = Py.strip(settings.verify()).lowercase()
        if (kind.isEmpty()) return null
        if (kind == "same") return provider
        return try {
            makeProvider(
                LLMSettings(provider = kind, model = settings.verifyModel()),
                timeoutS = 60.0,
                keys = KeyStore { name -> settings.secret(name) },
            )
        } catch (exc: Exception) {
            Log.w(TAG, "no verifier: $exc")
            null
        }
    }

    /**
     * The coach's model at the effort for mundane replies; the provider itself when that is
     * unset or the provider has no effort to set. `cli._routine_provider`.
     */
    private fun routineProvider(settings: SettingsStore, provider: LLMProvider): LLMProvider {
        val effort = settings.llm().routineEffort
        if (effort.isNullOrEmpty() || provider !is OpenAICompatProvider) return provider
        return provider.withEffort(effort)
    }

    companion object {
        private const val TAG = "sidecoach.coach"

        /**
         * One checked claim as the model is shown it: `f"{v.claim.text!r}: {v.result}, {v.reason}"`.
         * The text in Python's repr -- double quotes around "it's", a backslash escaped -- where
         * the phone had put it in single quotes whatever it held (the parity audit cm-01).
         */
        internal fun claimLine(verdict: Verdict): String =
            "${pyRepr(verdict.claim.text)}: ${verdict.result.slug}, ${verdict.reason}"
    }
}
