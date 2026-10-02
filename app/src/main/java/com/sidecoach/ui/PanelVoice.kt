package com.sidecoach.ui

import android.content.Context
import android.os.SystemClock
import com.sidecoach.analysis.Engine
import com.sidecoach.coach.Coach
import com.sidecoach.events.Trigger
import com.sidecoach.voice.KokoroTts
import com.sidecoach.voice.TtsProvider
import com.sidecoach.voice.Voices
import org.json.JSONObject
import java.io.File

/**
 * The panel's voice: `ui/panel.py`'s speaking side. Which voice, muted or not, the preview line
 * for the settings page, and -- the part that matters in a game -- what the voice does with each
 * coach message so it never talks about a position the board has left behind (`voice_action`,
 * `_speak`, `_stream_sink`, `_say_held_word`), with the time from a move to its first sound
 * (`_note_move` ... `_on_audio_starts`).
 */
object PanelVoice {
    /**
     * `panel.STALE_AFTER`: how many plies behind the board a message of each trigger may still
     * be spoken; `STALE_PLIES` the fallback for any other.
     */
    private val STALE_AFTER = mapOf("post_move" to 2, "post_mistake" to 2, "decisive_question" to 0, "stuck" to 0)
    private const val STALE_PLIES = 2

    /** What the voice was last given, when a room (a review, a puzzle) gave it (`ROOM_VOICE`). */
    const val ROOM_VOICE = "room"

    /**
     * A "verdict" this short (post-move praise alone, "Good.") is skipped when the voice is
     * busy; one with a lesson in it is longer, and waits its turn.
     */
    private const val SHORT_VERDICT = 40

    /**
     * A fast game: the coach's words are shown, not spoken. The last unspoken word is said after
     * all once the player has been thinking in that position this long (Tai, 2026-09-22).
     */
    private const val PACE_RESUME_S = 6.0

    const val VOICE_SAMPLE =
        "That's the move -- rooks belong on open files. Now, what is your opponent's most " +
            "dangerous idea?"

    /** The voice, once built (`panel._tts`); null until the warm-up makes it. */
    @Volatile var tts: KokoroTts? = null
        private set

    /**
     * A voice a test speaks through instead of [tts], as the desktop's tests set `engine._tts`
     * to a fake: what the policy did is then seen without a model to speak it.
     */
    @Volatile internal var speaker: TtsProvider? = null

    /** What the panel speaks through: `self._tts`, the stand-in or the voice the warm-up built. */
    internal fun voiceOut(): TtsProvider? = speaker ?: tts

    @Volatile var muted = false
        private set

    @Volatile var voice = "af_sarah"
        private set

    /** The coach whose pace decides a fast game (`panel._coaching`), and the loop's clock. */
    @Volatile var coach: Coach? = null
    @Volatile var loopTs = 0.0

    /** What is being spoken now: its trigger and ply (`panel._voicing`). */
    @Volatile internal var voicing: Pair<String, Int?>? = null

    /** Counts the times the voice was cut off by the player (a barge-in, a mute): a reply being
     * written sentence by sentence says nothing more once it moves (`panel._hushed`). */
    @Volatile internal var hushed = 0

    /** A word held back for pace (`panel._held_word`): trigger, text, ply, since. */
    private data class Held(val trigger: String, val text: String, val ply: Int?, val since: Double)
    @Volatile private var heldWord: Held? = null

    private val lock = Any()
    private val moveAt = HashMap<Int, Double>()
    private val awaitingAudio = HashMap<String, Double>()
    private val latSpeech = doubleArrayOf(0.0, 0.0, 0.0)
    private val latToken = doubleArrayOf(0.0, 0.0, 0.0)

    fun modelDir(context: Context): File = File(context.filesDir, "models/kokoro")

    /**
     * The voice built from the settings (`panel._build_tts`), not loaded -- the warm-up loads it
     * off the main thread -- and the page told what there is: the voice, the voices on offer
     * and where it stands.
     */
    fun build(context: Context, settings: SettingsStore) {
        val made = KokoroTts(context.applicationContext, modelDir(context), settings.voice(), settings.speed(), settings.lang())
        made.onPlay = { line -> onAudioStarts(line) }
        made.onSynthesis = { synthesising -> juggle(synthesising) }
        // The words are shown when the coach has them (`speak`); the voice, when it comes,
        // keeps them up while it speaks.
        made.onCaption = { _, speechMs -> Captions.voiced(speechMs) }
        // The voice it replaces is cut off and freed (off this thread: a line being
        // synthesised finishes first).
        tts?.let { old ->
            runCatching { old.stop() }
            kotlin.concurrent.thread(name = "voice-close", isDaemon = true) { old.close() }
        }
        tts = made
        voice = settings.voice()
        PanelEngine.set(
            "voice" to voice,
            "voices" to Voices.english(made.voicesPath),
            "muted" to muted,
            "voice_detail" to made.healthcheck().second,
        )
    }

    /** The engine held paused while a piece of a line is made, to let go of after. */
    @Volatile private var pausedEngine: Engine? = null

    /**
     * The phone's own (Tai, 2026-09-23): Kokoro or Stockfish at work, not both. They share the
     * phone's big cores, and with both at work each runs at half speed, so the game's engine
     * is paused (never ended: its hash is kept) while a piece is made, and runs on after.
     */
    private fun juggle(synthesising: Boolean) {
        if (synthesising) {
            val engine = PanelWarm.assessing?.analyser?.annotator?.engine ?: return
            pausedEngine = engine
            engine.pause()
        } else {
            pausedEngine?.resume()
            pausedEngine = null
        }
    }

    /** Change the speaking voice: a style on the loaded model, nothing reloads. `set_voice`. */
    fun setVoice(name: String) {
        tts?.voice = name
        voice = name
        val detail = tts?.healthcheck()?.second ?: "voice off"
        PanelEngine.set("voice" to name, "voice_detail" to detail)
        PanelEngine.emit("voice", "info", "voice set to $name", JSONObject().put("detail", detail))
    }

    /**
     * Muted from the panel or the caption box's own button, and kept for the next session:
     * `/api/muted`, which sets it and remembers it (`settings_api.remember`). One press at a
     * time: two quick ones, each on its own thread, saved out of order left the voice unmuted
     * and the setting muted. Under the settings' own lock, so a Settings save at the same moment
     * cannot put it back either (the settings review, 2026-10-01).
     */
    fun mute(settings: SettingsApi, on: Boolean) {
        synchronized(SettingsApi.SAVING) {
            setMuted(on)
            settings.remember(JSONObject().put("voice", JSONObject().put("muted", on)))
        }
    }

    /** Muted as the last session was left, at the warm-up, with no word in the log
     * (`self.muted = self.muted or self.settings.voice.muted`). */
    fun startMuted(on: Boolean) {
        muted = on
        PanelEngine.set("muted" to on)
        Captions.muted(on)
    }

    /**
     * A room's words (a review's card, a puzzle's coach) in the panel's voice, captioned as the
     * panel's are: `server._room_voice`'s speak. Mute is silence everywhere: the coach's Mute
     * silenced the coach only, and Game Review and the puzzles talked on (voice finding 12).
     */
    fun roomSay(text: String) {
        val speaker = voiceOut() ?: return
        Captions.show(text)
        if (muted) return
        voicing = ROOM_VOICE to null
        speaker.say(text)
    }

    /** A room's hush: only a room's own words; closing a review cut off the live coach. */
    fun roomStop() {
        if (voicing?.first == ROOM_VOICE) voiceOut()?.stop()
    }

    /**
     * The position the coach is speaking about is left -- a new game, a takeback, the board
     * re-read, the player's own next move -- so what the voice is saying about it stops (Tai,
     * 2026-10-02: "interrupt the tts so it stops its long rant"). The same mechanism as the
     * push-to-talk silencing and the barge-in: the voice stops, and the rest of a reply still
     * being written goes to the captions and the record, silent ([streamSink]). The text is never
     * cut. The opponent's move alone does not stop it. `panel._position_moved_on`.
     */
    fun positionMovedOn(why: String) {
        hushed += 1
        val speaker = voiceOut() ?: return
        if (speaker.busy) {
            speaker.stop()
            PanelEngine.emit("voice", "debug", "cut off: $why")
        }
    }

    /** `set_muted`. */
    fun setMuted(on: Boolean) {
        muted = on
        PanelEngine.set("muted" to on)
        Captions.muted(on)
        if (on) {
            hushed += 1
            voiceOut()?.stop()
        }
        PanelEngine.emit("voice", "info", if (on) "muted" else "unmuted")
    }

    /**
     * One line in [name] (the coach's own when null), for the settings page's Listen button;
     * the coach's voice is unchanged. False when the voice cannot speak. `say_sample`.
     */
    fun saySample(name: String?, text: String = VOICE_SAMPLE): Boolean {
        val speaker = voiceOut()
        val (ready, detail) = speaker?.healthcheck() ?: (false to "voice off")
        if (!ready || speaker == null) {
            PanelEngine.emit("voice", "warn", "preview: the voice cannot speak: $detail")
            return false
        }
        speaker.stop()
        speaker.say(text, voice = name?.ifEmpty { null })
        PanelEngine.emit("voice", "info", "preview in ${Voices.label(name?.ifEmpty { null } ?: voice)}: $text")
        return true
    }

    // ---- what the voice does with a coach message --------------------------------------------

    /**
     * What the voice should do with a coach message, so it never talks about a position the
     * board has left behind. `PanelEngine.voice_action`, line for line: a message further
     * behind the board than its trigger allows is shown but not spoken ("stale"); a short
     * post-move verdict is dropped if the voice is busy ("skip"); anything substantive cuts off
     * what is still playing when that is about a moment the board has left ("supersede") and
     * waits its turn behind it otherwise ("speak"). The answer to the user's own question is
     * never stale and never waits: they asked.
     */
    fun voiceAction(
        trigger: String,
        text: String,
        ply: Int?,
        plyNow: Int,
        busy: Boolean,
        playing: Pair<String, Int?>? = null,
        fast: Boolean = false,
    ): String {
        if (trigger == "user_ask") return if (busy) "supersede" else "speak"
        if (fast && trigger != "review") return "text" // a fast game: on the page now
        val allowed = STALE_AFTER[trigger] ?: (STALE_PLIES - 1)
        // The review sums up a game that is over: the board has nowhere to leave it for. Its
        // message carries the latest position with facts, which after a mate is a move or two
        // behind the board, and must not count as behind.
        if (ply != null && trigger != "review" && plyNow - ply > allowed) return "stale"
        if (trigger == "post_move" && text.length < SHORT_VERDICT) return if (busy) "skip" else "speak"
        if (!busy) return "speak"
        if (playing == null) return "supersede"
        val (oldTrigger, oldPly) = playing
        val oldAllowed = STALE_AFTER[oldTrigger] ?: (STALE_PLIES - 1)
        val oldStale = oldPly != null && plyNow - oldPly > oldAllowed
        return if (oldStale) "supersede" else "speak"
    }

    /** The game is fast and the player has not slowed down in this position. `_coach_fast`. */
    private fun coachFast(): Boolean = coach?.fastNow(loopTs) == true

    private fun plyNow(): Int = PanelEngine.plyNow()

    /** Apply [voiceAction] and speak accordingly; what was done, for the log. `_speak`. */
    fun speak(trigger: String, text: String, ply: Int?, movedOn: Boolean = false): String {
        // The captions show every line whatever the voice does with it -- muted, skipped as
        // short, stale, held back for pace (Tai, 2026-10-01). While the push-to-talk key is held
        // the voice says nothing: the player is talking. A reply asked for a moment that is over
        // ([movedOn]: a new game, a takeback, a re-read, the player's next move) is shown in its
        // place and not spoken (Tai, 2026-10-02).
        if (muted || PanelEngine.pttHeld()) return "muted".also { Captions.show(text) }
        val speaker = voiceOut() ?: return "muted".also { Captions.show(text) }
        if (movedOn) return "stale".also { Captions.show(text) }
        val action = voiceAction(trigger, text, ply, plyNow(), speaker.busy, voicing, coachFast())
        if (action == "supersede") speaker.stop()
        if (action == "speak" || action == "supersede") {
            voicing = trigger to ply
            awaitAudio(trigger, ply, text)
            Captions.show(text)  // at once; the voice follows when it is made
            speaker.say(text)
            PanelEngine.emit("voice", "debug", "speaking ($voice): ${PanelEngine.elide(text, 60)}")
        } else if (action == "text") {
            heldWord = Held(trigger, text, ply, SystemClock.elapsedRealtime() / 1000.0)
            Captions.show(text)
        } else {
            Captions.show(text) // "skip", "stale": not said, and shown all the same
        }
        return action
    }

    /**
     * The voice's sink for one coach reply written sentence by sentence (`_stream_sink`): the
     * first sentence decides, by [voiceAction], whether the reply is spoken at all, and each
     * sentence is queued as it comes, so the first is heard while the model writes the rest. A
     * reply that has begun is heard to its end unless the position it is about is left: a new
     * game, a takeback, a re-read or the player's next move cuts it ([positionMovedOn]); the
     * opponent's move alone does not. Called on the coach's thread.
     *
     * No reply is dropped (Tai, 2026-10-02): a reply that is not to be spoken -- stale, skipped, or
     * asked for a moment that is over ([movedOn]) -- is taken sentence by sentence all the same,
     * captioned and kept as the reply's text, and only silent. Declining it, as before, made the
     * coach say a template in its place.
     */
    fun streamSink(trigger: Trigger, ply: Int?, movedOn: () -> Boolean = { false }): ((String) -> Boolean)? {
        // No voice yet (the warm-up has not built it): nothing to stream to, as when muted -- the
        // reply is read whole and shown. A sink that declined here made the coach say its template
        // in place of every AI reply until the voice was built (the round-3 parity audit).
        if (muted || voiceOut() == null) return null
        val name = trigger.slug
        var started = false
        var silent = false // began while the push-to-talk key was held, or is not to be spoken
        var hushedAt = hushed
        return sink@{ sentence ->
            val continues = started
            if (started && (muted || hushed != hushedAt || silent)) {
                // Muted, or cut off (barge-in, hush) mid-reply: the rest of the reply is still
                // the reply -- in the chat, the record and the captions -- only silent.
                // Declining here cut the reply's text at what had been said.
                Captions.show(sentence, true)
                return@sink true
            }
            if (muted) return@sink false
            if (!started && PanelEngine.pttHeld()) {
                // A reply that begins while the player holds the key to talk is never spoken,
                // however long the key is held: captioned, in the chat, silent.
                started = true
                silent = true
                Captions.show(sentence)
                return@sink true
            }
            val speaker = voiceOut() ?: run {
                // The voice went away mid-reply: the rest is still the reply, only silent.
                started = true
                silent = true
                Captions.show(sentence, continues)
                return@sink true
            }
            if (!started) {
                hushedAt = hushed
                val action = if (movedOn()) {
                    "stale"
                } else {
                    voiceAction(name, sentence, ply, plyNow(), speaker.busy, voicing, coachFast())
                }
                if (action == "supersede") speaker.stop()
                if (action == "text") return@sink false // a fast game: held for the pace, not lost
                if (action != "speak" && action != "supersede") {
                    // Shown, in its place, not spoken: the rest of the reply follows the same.
                    started = true
                    silent = true
                    Captions.show(sentence)
                    PanelEngine.emit("voice", "debug", "not spoken ($action): ${PanelEngine.elide(sentence, 60)}")
                    return@sink true
                }
                started = true
                voicing = name to ply
                awaitAudio(name, ply, sentence)
            }
            // Each sentence as the model writes it, the reply's later ones run on from the first;
            // its voice follows.
            Captions.show(sentence, continues)
            speaker.say(sentence)
            PanelEngine.emit("voice", "debug", "speaking ($voice): ${PanelEngine.elide(sentence, 60)}")
            true
        }
    }

    /**
     * A word held back for pace is spoken once the player has been thinking in that position for
     * [PACE_RESUME_S]: the game slowed down. A word about a position the board has left is
     * dropped. [ts] is the loop's clock. `_say_held_word`.
     */
    fun sayHeldWord(coach: Coach?, ts: Double) {
        val held = heldWord ?: return
        if (muted || PanelEngine.pttHeld()) return
        val plyNow = plyNow()
        if (held.ply != null && plyNow - held.ply > (STALE_AFTER[held.trigger] ?: (STALE_PLIES - 1))) {
            heldWord = null
            return
        }
        if (coach == null || coach.finished) {
            heldWord = null // the game is over: the review speaks, not a late verdict
            return
        }
        if (!coach.policy.userToMove) return
        // Thinking in this position for PACE_RESUME_S -- counted from when the position
        // appeared, not from when the word arrived.
        val began = coach.policy.positionSince
        val thinking = if (began != null) ts - began else SystemClock.elapsedRealtime() / 1000.0 - held.since
        val speaker = voiceOut() ?: return
        if (thinking < PACE_RESUME_S || speaker.busy) return
        heldWord = null
        voicing = held.trigger to held.ply
        awaitAudio(held.trigger, held.ply, held.text)
        Captions.again() // captioned when it was held back, long faded: back while it is said
        speaker.say(held.text)
        PanelEngine.emit("voice", "debug", "speaking ($voice), the game slowed: ${PanelEngine.elide(held.text, 60)}")
    }

    // ---- latency, amortised: the move to the first sound, the model's first token -----------

    /** A move was committed now (the panel's clock, not the frame's). `_note_move`. */
    fun noteMove(ply: Int) {
        synchronized(lock) {
            moveAt[ply] = SystemClock.elapsedRealtime() / 1000.0
            // Only the last few matter; a move nobody spoke about is forgotten.
            moveAt.keys.filter { it < ply - 4 }.forEach { moveAt.remove(it) }
        }
    }

    /**
     * The first line of a word on a move is about to be queued: which move, so its first sound
     * can be timed. The message's ply is the position the move was played from; the move itself
     * is the next ply. `_await_audio`.
     */
    private fun awaitAudio(trigger: String, ply: Int?, line: String) {
        if ((trigger != "post_move" && trigger != "post_mistake") || ply == null) return
        synchronized(lock) {
            val moved = moveAt.remove(ply + 1)
            if (moved != null) awaitingAudio[line] = moved
        }
    }

    /** The voice's thread: a line's audio is playing now. `_on_audio_starts`. */
    internal fun onAudioStarts(line: String) {
        val now = SystemClock.elapsedRealtime() / 1000.0
        synchronized(lock) {
            val moved = awaitingAudio.remove(line) ?: return
            val seconds = now - moved
            latSpeech[0] += 1.0
            latSpeech[1] += seconds
            latSpeech[2] = seconds
        }
    }

    fun noteFirstToken(seconds: Double) {
        synchronized(lock) {
            latToken[0] += 1.0
            latToken[1] += seconds
            latToken[2] = seconds
        }
    }

    /** `_latency`: the averages and the last, for the status strip. */
    fun latency(): Map<String, Any?> = synchronized(lock) {
        val (speechN, speechTotal, speechLast) = latSpeech
        val (tokenN, tokenTotal, tokenLast) = latToken
        mapOf(
            "move_to_speech_n" to speechN.toInt(),
            "move_to_speech_s" to if (speechN > 0) round2(speechTotal / speechN) else null,
            "move_to_speech_last_s" to if (speechN > 0) round2(speechLast) else null,
            "first_token_n" to tokenN.toInt(),
            "first_token_s" to if (tokenN > 0) round2(tokenTotal / tokenN) else null,
            "first_token_last_s" to if (tokenN > 0) round2(tokenLast) else null,
        )
    }

    private fun round2(value: Double): Double =
        java.math.BigDecimal(value).setScale(2, java.math.RoundingMode.HALF_EVEN).toDouble()
}
