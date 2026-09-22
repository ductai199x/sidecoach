package com.sidecoach.coach

import android.util.Log
import com.sidecoach.events.Event
import com.sidecoach.llm.KeyStore
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.makeProvider
import com.sidecoach.ui.PanelEngine
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
class CoachRunner(settings: SettingsStore) {

    val coach: Coach?
    var detail: String = "off"
        private set

    init {
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
        val model = provider!!
        val offline = model is OfflineProvider
        this.detail = "${model.name} ($detail)"
        coach = try {
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
        PanelEngine.emit("sys", if (offline) "warn" else "info", "coach: ${this.detail}")
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
        val coach = coach ?: return
        PanelEngine.heard(text)
        coach.ask(text, ts)
    }

    private var drained = 0

    /** Whatever the coach has finished saying, onto the panel, with what the checks found. */
    fun drain() {
        val coach = coach ?: return
        coach.poll() // records each reply as a turn of the conversation
        val spoken = coach.spoken
        while (drained < spoken.size) {
            val one = spoken[drained++]
            // A failed reply with nothing to fall back on is kept for the log, said to no one.
            if (one.message.text.isEmpty()) continue
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
