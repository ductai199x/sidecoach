package com.sidecoach.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.SystemClock
import com.sidecoach.chess.Board
import com.sidecoach.voice.Correct
import com.sidecoach.voice.Listener
import com.sidecoach.voice.MODES
import com.sidecoach.voice.Microphone
import com.sidecoach.voice.ParakeetStt
import com.sidecoach.voice.Py
import com.sidecoach.voice.Recogniser
import com.sidecoach.voice.SAMPLE_RATE
import com.sidecoach.voice.SileroGate
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * The panel's hearing: `ui/panel.py`'s listening side. The recogniser, the voice gate, the
 * microphone and the listener, built once (`__post_init__`); how the coach listens
 * (`set_listen`), push-to-talk (`ptt`), a clip treated as said (`hear`), the settings page's
 * microphone test (`set_hear_test`, `hear_status`), and the dictation: words said go into the
 * text box as they are recognised and are sent a second after the utterance ends, unless more
 * is said, which appends, or the person types, which makes them theirs (Tai, 2026-09-22).
 *
 * Two things are the phone's own. Android asks the person before an app may use the microphone,
 * so a mode that needs it asks first ([askPermission], answered in [permitted]). And an app may
 * only hear while it is in front or a foreground service of the microphone type holds it: the
 * microphone is open while listening is on and the panel is in front or a session holds it
 * ([visible], [sessionHolds]), and closed otherwise -- in the background it would record
 * silence, and in voice mode run the gate on it thirty times a second. There is no global
 * push-to-talk key on a phone: the page's hold-to-talk button (or Space, on a keyboard) is it.
 */
object PanelListen {

    /**
     * `panel.DICTATION_SEND_DELAY_S`: words said are sent as the ask this long after the
     * utterance ends, unless the person presses again or types.
     */
    const val DICTATION_SEND_DELAY_S = 1.0

    /** The delay in use: [DICTATION_SEND_DELAY_S], shortened by a test as the desktop's monkeypatches it. */
    @Volatile internal var sendDelayS = DICTATION_SEND_DELAY_S

    /** `panel.HEAR_TEST_TIMEOUT_S`: the test turns itself off this long after the page's last look. */
    const val HEAR_TEST_TIMEOUT_S = 20.0

    /** The recogniser (`panel._stt`), the microphone and the listener; null until built. */
    @Volatile var stt: Recogniser? = null
        internal set
    @Volatile var mic: Microphone? = null
        private set
    @Volatile var listener: Listener? = null
        private set
    private var gate: SileroGate? = null
    private lateinit var appContext: Context

    /** `panel._lock` for the dictation: its words so far and the timer that sends them. */
    private val lock = Any()
    private var dictationText = ""
    private var dictationTimer: ScheduledFuture<*>? = null
    private val timers = Executors.newSingleThreadScheduledExecutor { r ->
        Thread(r, "dictation").apply { isDaemon = true }
    }

    /** One partial recognition at a time (`_partial_busy`); a partial still running skips the next. */
    private val partials = Executors.newSingleThreadExecutor { r ->
        Thread(r, "stt-partial").apply { isDaemon = true }
    }
    private val partialBusy = AtomicBoolean(false)

    /** Bumped when an utterance ends, so a partial of it that finishes late is dropped. */
    @Volatile private var utteranceGen = 0

    // The settings page's live test: while on, it listens (voice activity) and what it hears
    // goes to `heardTest`, not to the coach; the mode it interrupted is restored after.
    @Volatile private var hearTestOn = false
    @Volatile private var hearTestMode = "off"
    @Volatile private var hearTestSeen = 0.0
    private val heardTest = ArrayDeque<JSONObject>()

    // The phone's rule for an open microphone.
    private val micLock = Any()
    @Volatile private var visible = true
    @Volatile private var sessionHolds = false

    /** Asks the activity for the microphone; the answer comes back through [permitted]. */
    @Volatile var askPermission: (() -> Unit)? = null
    @Volatile private var asking = false
    @Volatile private var pendingMode: String? = null

    fun modelDir(context: Context): File = File(context.filesDir, "models/parakeet-tdt_ctc-110m")
    fun gateFile(context: Context): File = File(context.filesDir, "models/silero/${SileroGate.FILE}")

    private fun now(): Double = SystemClock.elapsedRealtime() / 1000.0

    /**
     * The recogniser, the gate, the microphone and the listener, built once and kept for the
     * app's life (`__post_init__`); the warm-up loads the model. The page is told there is
     * something to listen with.
     */
    fun build(context: Context) {
        synchronized(micLock) {
            appContext = context.applicationContext
            if (listener != null) return
            val recogniser = ParakeetStt(appContext, modelDir(appContext))
            val made = SileroGate(gateFile(appContext))
            val input = Microphone(appContext)
            stt = recogniser
            gate = made
            mic = input
            listener = Listener(
                input,
                made,
                onUtterance = ::onUtterance,
                onSpeechStart = ::onSpeechStart,
                coachSpeaking = { PanelVoice.tts?.busy == true },
                onEvent = { level, text -> PanelEngine.emit("voice", level, text) },
                onPartial = ::onPartial,
                onDropped = ::onDropped,
            )
            PanelEngine.set("stt_detail" to recogniser.healthcheck().second, "stt_ready" to true)
        }
    }

    /**
     * A new recogniser or microphone (`_reload_recogniser`): listening pauses, both are rebuilt
     * -- the old model freed, which the phone must do itself -- the voice's warm-up runs again,
     * and listening resumes in the mode it was in.
     */
    fun reload(warm: () -> Unit) {
        val ear = listener ?: return
        val mode = ear.mode
        runCatching { ear.setMode("off") }
        val old = stt
        val recogniser = ParakeetStt(appContext, modelDir(appContext))
        stt = recogniser
        thread(name = "stt-close", isDaemon = true) { old?.close() }
        val input = Microphone(appContext)
        mic = input
        ear.mic = input
        val (ready, detail) = recogniser.healthcheck()
        PanelEngine.set("stt_detail" to detail, "stt_ready" to ready)
        PanelEngine.emit("voice", "info", "speech recognition reloading: $detail")
        thread(name = "stt-reload", isDaemon = true) { warm() }
        if (mode != "off") runCatching { setListen(mode) }
    }

    // ---- how the coach listens ---------------------------------------------------------------

    private fun granted(): Boolean =
        ::appContext.isInitialized &&
            appContext.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    /**
     * How the coach listens: off, push-to-talk, or voice activity. `set_listen`. A mode that
     * needs the microphone before the person has allowed it asks, and is set again once they
     * have; until then it is off, with the reason.
     */
    fun setListen(mode: String): String {
        if (mode !in MODES) throw IllegalArgumentException("listen mode must be one of $MODES")
        val ear = listener ?: return "off"
        if (mode != "off" && !granted()) {
            pendingMode = mode
            ask()
        }
        val detail: String
        synchronized(micLock) {
            detail = ear.setMode(mode)
            refreshMic()
        }
        PanelEngine.set("listen" to ear.mode, "ptt" to false)
        PanelEngine.emit("voice", "info", "listening: $detail")
        return ear.mode
    }

    private fun ask() {
        val request = askPermission ?: return
        if (asking) return
        asking = true
        request()
    }

    /** The person answered the microphone's permission: the mode that asked is set now. */
    fun permitted(granted: Boolean) {
        asking = false
        val mode = pendingMode
        pendingMode = null
        if (!granted) {
            PanelEngine.emit(
                "voice",
                "warn",
                "listening: off (the microphone was not allowed; the app's page in Android's settings can allow it)",
            )
            return
        }
        PanelEngine.emit("voice", "info", "the microphone is allowed")
        // Off the main thread: opening the input and starting the listener take a moment.
        if (mode != null) thread(name = "listen", isDaemon = true) { runCatching { setListen(mode) } }
    }

    /** The panel came in front, or went behind another app. */
    fun visible(on: Boolean) {
        visible = on
        synchronized(micLock) { refreshMic() }
    }

    /** A session holds the microphone (its foreground service has the microphone type), or no more. */
    fun sessionHolds(on: Boolean) {
        sessionHolds = on
        synchronized(micLock) { refreshMic() }
    }

    /**
     * The microphone open while listening is on and the app may hear -- in front, or held by a
     * session -- and closed otherwise; the listener waits on it either way. Under [micLock].
     */
    private fun refreshMic() {
        val ear = listener ?: return
        val input = mic ?: return
        val wanted = ear.mode != "off" && (visible || sessionHolds)
        if (!wanted && input.isOpen) input.close() else if (wanted && !input.isOpen) input.open()
    }

    /** Push-to-talk pressed or released. `ptt`. */
    fun ptt(down: Boolean) {
        val ear = listener ?: return
        if (down) {
            if (ear.mode == "ptt") dictationHold()
            ear.press()
        } else {
            ear.release()
        }
        PanelEngine.set("ptt" to (down && ear.mode == "ptt"))
    }

    /** Treat a float32 16 kHz clip as something the user said (the test and dev path). `hear`. */
    fun hear(audio: FloatArray, seconds: Double? = null) {
        onUtterance(audio, seconds ?: (audio.size / SAMPLE_RATE.toDouble()))
    }

    // ---- the settings page's microphone test --------------------------------------------------

    /** Start or stop the live test of the microphone and speech recognition. `set_hear_test`. */
    fun setHearTest(on: Boolean): JSONObject {
        val ear = listener
        val input = mic
        val recogniser = stt
        if (ear == null || input == null || recogniser == null) {
            return JSONObject().put("ok", false).put("error", "speech recognition: not built yet")
        }
        if (on && !hearTestOn) {
            val (ok, micDetail) = input.healthcheck()
            val (ready, sttDetail) = recogniser.healthcheck()
            if (!ok || !ready) {
                // The phone's permission: asked now, and Start again once it is given.
                if (!ok && !granted()) ask()
                val what = if (!ok) "microphone" else "speech recognition"
                return JSONObject().put("ok", false).put("error", "$what: ${if (!ok) micDetail else sttDetail}")
            }
            hearTestMode = ear.mode
            synchronized(heardTest) { heardTest.clear() }
            hearTestOn = true
            hearTestSeen = now()
            runCatching {
                synchronized(micLock) {
                    ear.setMode("vad")
                    refreshMic()
                }
            }
            // The recogniser loads on its first utterance otherwise, and the first result would
            // carry the load in its time.
            if (!recogniser.loaded) thread(name = "stt-warm", isDaemon = true) { recogniser.load() }
            PanelEngine.emit("voice", "info", "microphone test on: say something")
        } else if (!on && hearTestOn) {
            hearTestOn = false
            runCatching { setListen(hearTestMode) }
            PanelEngine.emit("voice", "info", "microphone test off")
        }
        return hearStatus()
    }

    private fun noteHeardTest(raw: String, seconds: Double, ms: Double) {
        val heard = if (raw.isNotEmpty()) Correct.correct(raw, boardOnScreen()).text else ""
        synchronized(heardTest) {
            heardTest.addFirst(
                JSONObject()
                    .put("raw", raw)
                    .put("heard", heard)
                    .put("seconds", round(seconds, 1))
                    .put("stt_ms", roundInt(ms))
                    .put("t", System.currentTimeMillis() / 1000.0),
            )
            while (heardTest.size > 6) heardTest.removeLast()
        }
        PanelEngine.emit(
            "voice",
            "info",
            "microphone test heard: ${repr(raw)}" + (if (heard != raw) " -> ${repr(heard)}" else ""),
        )
    }

    /** What the page shows while the test is on: the level and what was heard. `hear_status`. */
    fun hearStatus(): JSONObject {
        hearTestSeen = now()
        val heard = synchronized(heardTest) { JSONArray(heardTest.toList()) }
        return JSONObject()
            .put("ok", true)
            .put("on", hearTestOn)
            .put("level", round(minOf(1.0, listener?.level ?: 0.0), 3))
            .put("heard", heard)
            .put("microphone", mic?.healthcheck()?.second ?: "not built yet")
            .put("speech_recognition", stt?.healthcheck()?.second ?: "not built yet")
    }

    // ---- what was said ------------------------------------------------------------------------

    /** The board the screen shows, for correcting what was heard, or null. `_board_on_screen`. */
    private fun boardOnScreen(): Board? {
        val fen = PanelEngine.fen() ?: return null
        return try {
            Board(fen)
        } catch (_: Exception) {
            null
        }
    }

    /**
     * The utterance so far, on the listener thread: recognised off it, one at a time (a partial
     * still running means this one is skipped, the next will catch up). `_on_partial`.
     */
    internal fun onPartial(audio: FloatArray) {
        if (hearTestOn || !partialBusy.compareAndSet(false, true)) return
        val gen = utteranceGen
        partials.execute {
            try {
                val raw = stt?.transcribe(audio) ?: ""
                val text = if (raw.isNotEmpty()) Correct.correct(raw, boardOnScreen()).text else ""
                // Dropped when its utterance ended meanwhile: the final result has it.
                if (text.isNotEmpty() && gen == utteranceGen) {
                    synchronized(lock) {
                        val joined = Py.strip("$dictationText $text")
                        PanelEngine.set("dictation" to dictation(joined, live = true, sendsAt = null))
                    }
                }
            } catch (exc: Exception) { // a partial is best effort; the final result still comes
                PanelEngine.emit("voice", "debug", "partial recognition failed: ${exc.message ?: ""}")
            } finally {
                partialBusy.set(false)
            }
        }
    }

    private fun dictation(text: String, live: Boolean, sendsAt: Long?): Map<String, Any?> =
        mapOf("text" to text, "live" to live, "sends_at" to sendsAt)

    /** (Re)start the second of quiet after which the dictation is sent. `_dictation_arm`. */
    private fun dictationArm() {
        val delay = (sendDelayS * 1000).toLong()
        synchronized(lock) {
            dictationTimer?.cancel(false)
            dictationTimer = timers.schedule({ dictationSend() }, delay, TimeUnit.MILLISECONDS)
            PanelEngine.set(
                "dictation" to dictation(dictationText, live = false, sendsAt = System.currentTimeMillis() + delay),
            )
        }
    }

    /**
     * An utterance that came to nothing: earlier words (if any) go on to send, else the box
     * stops saying it is listening. `_dictation_settle`.
     */
    private fun dictationSettle() {
        val pending = synchronized(lock) { dictationText.isNotEmpty() }
        if (pending) {
            dictationArm()
        } else {
            synchronized(lock) { PanelEngine.set("dictation" to dictation("", live = false, sendsAt = null)) }
        }
    }

    /** A new press or a new utterance: whatever was about to send waits for the rest. */
    internal fun dictationHold() {
        synchronized(lock) {
            dictationTimer?.cancel(false)
            dictationTimer = null
            PanelEngine.set("dictation" to dictation(dictationText, live = true, sendsAt = null))
        }
    }

    private fun dictationSend() {
        val text: String
        synchronized(lock) {
            text = dictationText
            dictationText = ""
            dictationTimer = null
            PanelEngine.set("dictation" to dictation("", live = false, sendsAt = null))
        }
        if (text.isNotEmpty()) PanelEngine.ask(text, spoken = true)
    }

    /** The person typed: the words stay theirs in the box, nothing sends on its own. */
    fun cancelDictation(): JSONObject {
        val text: String
        synchronized(lock) {
            dictationTimer?.cancel(false)
            dictationTimer = null
            text = dictationText
            dictationText = ""
            PanelEngine.set("dictation" to dictation("", live = false, sendsAt = null))
        }
        return JSONObject().put("ok", true).put("text", text)
    }

    /**
     * The user started talking: the coach stops mid-sentence (barge-in), and a dictation about
     * to send waits for what is coming. `_on_speech_start`.
     */
    internal fun onSpeechStart() {
        if (!hearTestOn) dictationHold()
        val voice = PanelVoice.tts
        if (voice != null && voice.busy) {
            voice.stop()
            PanelEngine.emit("voice", "debug", "barge-in: the coach was cut off")
        }
    }

    /**
     * An utterance began and came to nothing (a click, a tap too short to hold a word): a
     * dictation it held goes on to send, or the box stops saying it is listening. `_on_dropped`.
     */
    internal fun onDropped() {
        if (!hearTestOn) dictationSettle()
    }

    /**
     * Something was said: transcribe it, correct it against the board, and into the text box
     * after whatever was said before in this dictation, sent after a second of quiet unless more
     * comes. Runs on the listener's delivery thread (or the request's, for `/api/hear`).
     * `_on_utterance`.
     */
    internal fun onUtterance(audio: FloatArray, seconds: Double) {
        utteranceGen += 1
        val began = System.nanoTime()
        val raw = stt?.transcribe(audio) ?: ""
        val ms = (System.nanoTime() - began) / 1e6
        if (hearTestOn && now() - hearTestSeen > HEAR_TEST_TIMEOUT_S) setHearTest(false) // the page went away
        if (hearTestOn) {
            noteHeardTest(raw, seconds, ms)
            return
        }
        if (raw.isEmpty()) {
            PanelEngine.emit(
                "voice",
                "debug",
                "heard nothing in ${format1(seconds)}s of audio",
                JSONObject().put("stt_ms", roundInt(ms)),
            )
            dictationSettle()
            return
        }
        val fixed = Correct.correct(raw, boardOnScreen())
        val text = fixed.text
        if (text.isEmpty()) {
            dictationSettle()
            return
        }
        synchronized(lock) { dictationText = Py.strip("$dictationText $text") }
        PanelEngine.set("heard" to text)
        PanelEngine.emit(
            "voice",
            "debug",
            "heard: $text",
            JSONObject()
                .put("raw", raw)
                .put("moves", JSONArray(fixed.moves))
                .put("seconds", round(seconds, 2))
                .put("stt_ms", roundInt(ms)),
        )
        dictationArm()
    }

    // ---- the phone's health strip -------------------------------------------------------------

    /** Where the recogniser runs once loaded (`stt_device`), and the frames the microphone dropped. */
    fun sttDevice(): String? = if (stt?.loaded == true) "cpu" else null
    fun micDropped(): Int = mic?.dropped ?: 0

    // ---- Python's formatting -------------------------------------------------------------------

    /** `round(x, places)`: half to even. */
    private fun round(value: Double, places: Int): Double =
        BigDecimal(value).setScale(places, RoundingMode.HALF_EVEN).toDouble()

    /** `round(x)`: an int, half to even. */
    private fun roundInt(value: Double): Long =
        BigDecimal(value).setScale(0, RoundingMode.HALF_EVEN).toLong()

    /** `f"{x:.1f}"`. */
    private fun format1(value: Double): String =
        BigDecimal(value).setScale(1, RoundingMode.HALF_EVEN).toPlainString()

    /** A string as Python's `repr` writes it, for the log's `{raw!r}`. */
    private fun repr(text: String): String {
        val quote = if ('\'' in text && '"' !in text) '"' else '\''
        val out = StringBuilder().append(quote)
        for (ch in text) {
            when {
                ch == '\\' -> out.append("\\\\")
                ch == quote -> out.append('\\').append(ch)
                ch == '\n' -> out.append("\\n")
                ch == '\r' -> out.append("\\r")
                ch == '\t' -> out.append("\\t")
                ch.code < 0x20 || ch.code == 0x7f -> out.append("\\x%02x".format(ch.code))
                else -> out.append(ch)
            }
        }
        return out.append(quote).toString()
    }
}
