package com.sidecoach.coach

import android.util.Log
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.Event
import com.sidecoach.llm.KeyStore
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.makeProvider
import com.sidecoach.profile.Ask
import com.sidecoach.profile.Store
import com.sidecoach.ui.PanelEngine
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
 * The engine for what the player names is not here. The desktop starts a second Stockfish
 * for it; on a phone that is a second process competing for the same few cores and the same
 * thermal budget, and it serves one kind of question. Without one the coach still examines
 * the pieces and squares the player names -- only a line needs a search.
 */
class CoachRunner(settings: SettingsStore, enabled: Boolean = true) {

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
                draftCheck = { text, board, after ->
                    checkDraft(text, board, after).map { "${it.text}: ${it.reason}" }
                },
                claims = if (offline) null else claimsCheck(model),
                verifier = if (offline) null else verifier(settings, model),
                engine = null,
                summariseEvery = settings.summariseEvery(),
                summariser = model,
                compactRecord = settings.compactRecord(),
                routineProvider = routineProvider(settings, model),
                verbalizedSampling = settings.verbalizedSampling(),
                localVerdicts = settings.localVerdicts(),
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
            claims = if (offline) null else claimsCheck(provider),
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
                PanelEngine.emit("sys", "warn", "coach error: $exc")
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

    /** Whatever the coach has finished saying, onto the panel, with what the checks found. */
    fun drain() {
        val coach = coach ?: return
        val sink = sink
        if (sink != null) {
            for (message in coach.poll()) sink(message)
            return
        }
        coach.poll() // records each reply as a turn of the conversation
        // What the coach looked up in the record for a question about the player, before
        // the reply it went into, as `_coach_meta` logs it.
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
            // A failed reply with nothing to fall back on is kept for the log, said to no one.
            if (one.message.text.isEmpty()) continue
            // Written down in the profile as the conversation (`panel._record_dialogue`).
            PanelEngine.recordDialogue(one.message)
            PanelEngine.said(one.message, one.problems, one.attempts)
        }
    }

    val pending: Int get() = coach?.speaker?.pending ?: 0

    fun close() {
        coach?.close()
    }

    /** The user's stated reasons, checked against the board. `cli._claims_check`. */
    private fun claimsCheck(provider: LLMProvider): ClaimsCheck = ClaimsCheck { text, board, facts ->
        // Reasoning off: a local model spends the whole budget thinking otherwise, and the
        // claims come out the same in a fifth of the time.
        val found = extractClaims(text, board, provider, thinking = false)
        verifyClaims(found, board, facts = facts).map {
            "'${it.claim.text}': ${it.result.slug}, ${it.reason}"
        }
    }

    /** The verifier the settings ask for: none, the coach's own model, or another. */
    private fun verifier(settings: SettingsStore, provider: LLMProvider): LLMProvider? {
        val kind = settings.verify().trim().lowercase()
        if (kind.isEmpty()) return null
        if (kind == "same") return provider
        return try {
            makeProvider(
                settings.llm().copy(provider = kind),
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
    }
}
