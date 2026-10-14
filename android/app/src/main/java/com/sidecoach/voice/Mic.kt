package com.sidecoach.voice

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioDeviceInfo
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.SystemClock
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.BlockingQueue
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * The microphone and the listener: turning what the user says into utterances to transcribe.
 * `sidecoach/voice/mic.py`.
 *
 * [Microphone] reads 16 kHz mono frames into a bounded queue -- if nobody drains it, the oldest
 * frames are dropped, so it can never grow. [Listener] consumes those frames on its own thread in
 * one of two modes:
 *
 *   push-to-talk  frames are kept between [Listener.press] and [Listener.release], then handed
 *                 over as one utterance -- no VAD involved, no false triggers;
 *   voice         Silero decides: an utterance starts after two confident frames (and the coach
 *                 is cut off -- barge-in), keeps a short pre-roll so the first syllable is not
 *                 clipped, and ends after 400 ms of silence. Utterances are capped at
 *                 [MAX_UTTERANCE_S] so a noisy room cannot buffer forever.
 *
 * While the coach's voice plays, the microphone is ignored (the echo guard): in voice mode nothing
 * starts and nothing is kept, and push-to-talk keeps no pre-roll of it. A held push-to-talk key is
 * heard as ever -- pressing it silences the voice.
 *
 * Everything degrades: no permission, no input, or no VAD model leaves the coach deaf with a
 * stated reason, never crashed.
 */
val MODES = listOf("off", "ptt", "vad")

/** 320 ms kept before a detected start, so "knight" is not "ight". */
const val PRE_ROLL_FRAMES = 10
/** The longest utterance (Tai, 2026-10-02: 60 s, was 20 s -- a long question asked on the held
 * key lost everything past 20 s). */
const val MAX_UTTERANCE_S = 60.0

/** How often a microphone whose input failed is tried again (`MIC_REOPEN_S`). */
const val MIC_REOPEN_S = 3.0

/** Ten seconds of backlog before the oldest is dropped. */
const val QUEUE_FRAMES = SAMPLE_RATE / HOP * 10

/**
 * What the listener reads frames from: the [Microphone], or a test's own. The desktop's tests
 * hand the listener any object with these members; Kotlin needs the shape named.
 */
interface FrameSource {
    val frames: BlockingQueue<FloatArray>
    val detail: String
    val isOpen: Boolean
    fun open(): Boolean
    fun close()

    /** Open, but no frame will come again: the input failed (the device gone, the audio server
     * restarted). `Microphone.stalled`. */
    fun stalled(): Boolean = false
}

/** What decides a frame is speech: the [SileroGate], or a test's script. */
interface VoiceGate {
    fun healthcheck(): Pair<Boolean, String>
    fun load(): Boolean
    fun reset()
    fun probability(input: FloatArray): Float
}

/** Speech to text, loaded once and kept: `stt.STTProvider`, as [ParakeetStt] provides it. */
interface Recogniser {
    val loaded: Boolean
    val loadSeconds: Double?
    fun healthcheck(): Pair<Boolean, String>
    fun load(): Boolean
    fun prime(): Double?
    fun transcribe(audio: FloatArray): String
    fun close()
}

/** 16 kHz mono frames from the phone's microphone (or the headset's), into a bounded queue. */
class Microphone(private val context: Context) : FrameSource {
    override val frames: BlockingQueue<FloatArray> = ArrayBlockingQueue(QUEUE_FRAMES)

    @Volatile var dropped = 0
        private set

    @Volatile override var detail = "not open"
        private set

    @Volatile private var record: AudioRecord? = null
    private var reader: Thread? = null
    @Volatile private var reading = false

    /**
     * Whether there is an input to open, and which: the phone's own default -- its microphone,
     * or a headset's while one is connected -- as the desktop follows the system default. The
     * one thing a phone adds is the person's permission.
     */
    fun healthcheck(): Pair<Boolean, String> {
        val granted = context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        if (!granted) return false to "no permission to use the microphone yet"
        return try {
            val audio = context.getSystemService(AudioManager::class.java)
            val inputs = audio.getDevices(AudioManager.GET_DEVICES_INPUTS)
            if (inputs.isEmpty()) return false to "no input device"
            val now = record?.routedDevice ?: inputs.firstOrNull { it.type == AudioDeviceInfo.TYPE_BUILTIN_MIC } ?: inputs[0]
            val name = now.productName?.toString()?.ifEmpty { null } ?: "the phone's microphone"
            true to "the system default input ($name now; follows a change when reopened)"
        } catch (exc: Exception) {
            false to "no input device: $exc"
        }
    }

    override val isOpen: Boolean get() = record != null

    override fun stalled(): Boolean = record != null && !reading

    override fun open(): Boolean {
        if (record != null) return true
        val (ok, why) = healthcheck()
        detail = why
        if (!ok) return false
        var made: AudioRecord? = null
        return try {
            val minimum = AudioRecord.getMinBufferSize(SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
            @Suppress("MissingPermission")
            made = AudioRecord(
                // The source Android tunes for recognisers: no gain riding, no noise gate.
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_FLOAT,
                maxOf(minimum, HOP * 4 * 8),
            )
            if (made.state != AudioRecord.STATE_INITIALIZED) {
                made.release()
                detail = "could not open input: the microphone did not initialise"
                return false
            }
            made.startRecording()
            record = made
            reading = true
            val source = made
            reader = Thread({ readLoop(source) }, "microphone").apply {
                isDaemon = true
                start()
            }
            true
        } catch (exc: Exception) {
            detail = "could not open input: $exc"
            runCatching { made?.release() }
            record = null
            false
        }
    }

    /**
     * The device's frames into the queue, the oldest dropped when nobody keeps up: what
     * sounddevice's callback does, on a thread of the phone's own. An input that fails (the
     * audio server restarted, the device gone) ends the loop rather than spinning on it.
     */
    private fun readLoop(source: AudioRecord) {
        while (reading) {
            val frame = FloatArray(HOP)
            var filled = 0
            while (filled < HOP && reading) {
                val n = source.read(frame, filled, HOP - filled, AudioRecord.READ_BLOCKING)
                if (n < 0) {
                    VoiceLog.warn("the microphone failed ($n); it stays closed until reopened")
                    detail = "the input failed ($n)"
                    reading = false
                }
                if (n <= 0) break
                filled += n
            }
            if (!reading) continue
            if (filled < HOP) {
                // A read that returned nothing: never spin on it.
                SystemClock.sleep(10)
                continue
            }
            if (!frames.offer(frame)) {
                // Nobody is listening fast enough: drop the oldest, never grow.
                dropped += 1
                frames.poll()
                frames.offer(frame)
            }
        }
    }

    override fun close() {
        val source = record
        record = null
        if (source != null) {
            reading = false
            runCatching { source.stop() }
            runCatching { reader?.join(500) }
            runCatching { source.release() }
            reader = null
        }
        frames.clear()
    }
}

/** Frames in, utterances out, in push-to-talk or voice mode. `mic.Listener`. */
class Listener(
    @Volatile var mic: FrameSource,
    val gate: VoiceGate,
    private val onUtterance: (FloatArray, Double) -> Unit,
    private val onSpeechStart: () -> Unit = {},
    private val coachSpeaking: () -> Boolean = { false },
    private val onEvent: (String, String) -> Unit = { _, _ -> },
    /**
     * While an utterance is being captured, the audio so far every [partialEveryS], for words on
     * the screen as they are said (Tai, 2026-09-22). Called on the listener thread.
     */
    private val onPartial: ((FloatArray) -> Unit)? = null,
    private val partialEveryS: Double = 0.4,
    /**
     * An utterance that began ([onSpeechStart] was called, or the key went down) and is not
     * handed over -- a click the endpointer dropped, a tap too short to hold a word, a mode
     * switch mid-sentence -- is said too, so whoever was told it began is told it came to
     * nothing (2026-09-23: a cough held a dictation's send for good).
     */
    private val onDropped: () -> Unit = {},
    /**
     * The listener's clock, in seconds (`mic.time.monotonic`): the device's uptime clock. A
     * test gives one it moves (the desktop's tests give the module a ManualClock in place of
     * its `time`); nothing else changes.
     */
    private val clock: () -> Double = { SystemClock.elapsedRealtime() / 1000.0 },
) {
    val endpointer = Endpointer()

    @Volatile var mode = "off"
        private set
    private var ptt = false

    /** The microphone's level, for a meter: the latest frame's peak, decaying. */
    @Volatile var level = 0.0
        private set
    private var buffer = ArrayList<FloatArray>()
    private val pre = ArrayDeque<FloatArray>()
    private var startedAt: Double? = null
    private var lastPartial = 0.0
    private var thread: Thread? = null
    @Volatile private var stop = false
    private val lock = Any()

    // Utterances are handed over on a thread of their own, so a recognition (hundreds of
    // milliseconds) never holds the listener's lock.
    private val deliveries = LinkedBlockingQueue<Pair<FloatArray, Double>>()
    private var deliverer: Thread? = null

    // Push-to-talk: the VAD hears the held frames on the listener thread, as the voice mode's
    // start does -- two frames in a row over its bar are speech; a press held over silence is
    // handed over as nothing (the recogniser made "yeah" of it, Tai 2026-10-14). [pttVad] is
    // whether a VAD is there to ask (null: not asked yet this press); without one, all that was
    // held is handed over.
    private var presses = 0
    private var pttChecked = 0
    private var pttRun = 0
    private var pttVad: Boolean? = null
    private var pttSpoke = false

    private fun now(): Double = clock()

    // ---- control ---------------------------------------------------------------------------

    /** Switch modes; opens the mic on demand and closes it when off. A detail, for the log. */
    fun setMode(wanted: String): String {
        if (wanted !in MODES) throw IllegalArgumentException("mode must be one of $MODES")
        synchronized(lock) {
            finish(discard = true)
            mode = wanted
            // A key held across the switch: the next press starts afresh (it was ignored).
            ptt = false
        }
        if (wanted == "off") {
            stop = true
            runCatching { thread?.join(2000) }
            thread = null
            mic.close()
            return "off"
        }
        if (!mic.open()) {
            mode = "off"
            return "off (${mic.detail})"
        }
        if (wanted == "vad" && !gate.load()) {
            mode = "ptt"
            onEvent("warn", "voice mode unavailable (${gate.healthcheck().second}); push-to-talk instead")
        }
        if (thread?.isAlive != true) {
            stop = false
            thread = Thread({ run() }, "listener").apply {
                isDaemon = true
                start()
            }
        }
        return "$mode (${mic.detail})"
    }

    /** For the tests: a push-to-talk press is being kept (the desktop's `_started_at`). */
    internal val pressTaken: Boolean get() = synchronized(lock) { ptt && startedAt != null }

    /** Push-to-talk down: start keeping frames. */
    fun press() {
        synchronized(lock) {
            if (mode != "ptt" || ptt) return
            ptt = true
            buffer = ArrayList(pre)
            startedAt = now()
            presses += 1
            pttChecked = 0
            pttRun = 0
            pttVad = null
            pttSpoke = false
        }
        onSpeechStart()
    }

    /** Push-to-talk up: what was kept is one utterance. */
    fun release() {
        synchronized(lock) {
            if (!ptt) return
            ptt = false
            finish(discard = false)
        }
    }

    private var reopenTried = 0.0

    /** The last frame was the coach's voice (the echo guard, [run]). */
    private var echoed = false

    /**
     * A microphone whose input failed is reopened, at most once every [MIC_REOPEN_S]: it stayed
     * closed until the mode was switched, and the coach deaf (voice finding 17).
     */
    private fun reopenIfStalled() {
        val t = now()
        if (t - reopenTried < MIC_REOPEN_S || !mic.stalled()) return
        reopenTried = t
        mic.close()
        if (mic.open()) onEvent("info", "microphone reopened (${mic.detail})")
    }

    // ---- the loop ----------------------------------------------------------------------------

    private fun run() {
        gate.reset()
        endpointer.reset()
        while (!stop) {
            val frame = mic.frames.poll(200, TimeUnit.MILLISECONDS)
            if (frame == null) {
                level *= 0.5
                reopenIfStalled()
                continue
            }
            var peak = 0.0
            for (v in frame) peak = maxOf(peak, kotlin.math.abs(v.toDouble()))
            level = maxOf(peak, level * 0.8)
            var partial: FloatArray? = null
            val current: String
            // The echo guard (Tai, 2026-10-02): while the coach's voice plays, what the
            // microphone hears is ignored -- it hears the speakers, and the coach's own words
            // were cutting it off and coming back as a question. A held push-to-talk key is the
            // exception (it silences the voice at once). Listening resumes with the first frame
            // after the voice stops.
            val echo = coachSpeaking()
            var unheard: List<FloatArray> = emptyList()
            var press = 0
            synchronized(lock) {
                current = mode
                if (current == "ptt") {
                    if (ptt) {
                        buffer.add(frame)
                        if (cap()) {
                            // A key held past the longest utterance: what was said goes to the
                            // recogniser, and holding on keeps nothing more until the next press.
                            // The buffer grew without end (voice finding 7).
                            ptt = false
                            finish(discard = false)
                        } else {
                            if (!pttSpoke) {
                                unheard = ArrayList(buffer.subList(pttChecked, buffer.size))
                                pttChecked = buffer.size
                                press = presses
                            }
                            // No words on the screen before any were said.
                            if (pttSpoke || pttVad == false) partial = partialDue()
                        }
                    } else if (echo) {
                        // Not held: the pre-roll a press would keep holds no coach's words.
                        pre.clear()
                    } else {
                        pre.addLast(frame)
                        while (pre.size > PRE_ROLL_FRAMES) pre.removeFirst()
                    }
                }
            }
            if (current != "vad") {
                if (unheard.isNotEmpty()) hearHeld(unheard, press)
                partial?.let { onPartial?.invoke(it) }
                continue
            }
            if (echo) {
                // Voice mode, the coach speaking: the frame counts as silence and is kept
                // nowhere. An utterance begun before the voice ends as the player's words alone;
                // none can begin. The VAD is not run on it (its state starts afresh with the
                // next frame heard).
                synchronized(lock) {
                    echoed = true
                    pre.clear()
                    when (endpointer.feed(0.0)) {
                        "end" -> finish(discard = false)
                        "drop" -> finish(discard = true)
                    }
                }
                continue
            }
            if (echoed) {
                echoed = false
                gate.reset()
            }
            // Voice mode: the VAD runs outside the lock; it is the slow part.
            val prob = gate.probability(frame).toDouble()
            var event: String?
            var skip = false
            synchronized(lock) {
                event = endpointer.feed(prob)
                if (endpointer.speaking) {
                    if (event == "start") {
                        buffer = ArrayList(pre)
                        startedAt = now()
                    }
                    buffer.add(frame)
                    if (cap()) {
                        endpointer.reset()
                        finish(discard = false)
                        skip = true
                    } else {
                        partial = partialDue()
                    }
                } else {
                    pre.addLast(frame)
                    while (pre.size > PRE_ROLL_FRAMES) pre.removeFirst()
                }
                if (!skip) {
                    when (event) {
                        "start" -> {
                            onEvent("debug", "speech started")
                            onSpeechStart()
                        }
                        "end" -> finish(discard = false)
                        "drop" -> {
                            finish(discard = true)
                            onEvent("debug", "a sound too short to be an ask, ignored")
                        }
                    }
                }
            }
            if (skip) continue
            partial?.let { onPartial?.invoke(it) }
        }
    }

    /**
     * The VAD on frames held under the push-to-talk key (outside the lock: it is the slow part),
     * until speech is heard: two frames in a row over the voice mode's start.
     */
    private fun hearHeld(frames: List<FloatArray>, press: Int) {
        var vad = synchronized(lock) { pttVad }
        if (vad == null) {
            vad = gate.load()
            gate.reset()
        }
        if (!vad) {
            synchronized(lock) { if (press == presses) pttVad = false }
            return
        }
        var run = pttRun
        for (frame in frames) {
            run = if (gate.probability(frame).toDouble() >= endpointer.start) run + 1 else 0
            if (run >= endpointer.startFrames) break
        }
        synchronized(lock) {
            if (press == presses) {
                pttVad = true
                pttRun = run
                pttSpoke = run >= endpointer.startFrames
            }
        }
    }

    /** True (and stop growing) once an utterance has reached its maximum length. */
    private fun cap(): Boolean = buffer.size * HOP >= MAX_UTTERANCE_S * SAMPLE_RATE

    /** The audio so far, when a partial is wanted and it is time for one (under the lock). */
    private fun partialDue(): FloatArray? {
        if (onPartial == null || buffer.size * HOP < 0.6 * SAMPLE_RATE) return null
        val now = now()
        if (now - lastPartial < partialEveryS) return null
        lastPartial = now
        return concat(buffer)
    }

    /** Hand the buffered frames over as one utterance (or drop them) and reset. */
    private fun finish(discard: Boolean) {
        val kept = buffer
        buffer = ArrayList()
        val started = startedAt
        startedAt = null
        pre.clear()
        gate.reset()
        val silent = mode == "ptt" && pttVad == true && !pttSpoke
        pttVad = null
        var drop = discard
        if (silent && kept.isNotEmpty() && !discard) {
            drop = true
            onEvent("debug", "push-to-talk held with nothing said, ignored")
        }
        val audio = if (kept.isNotEmpty() && !drop) concat(kept) else null
        val seconds = if (audio != null) audio.size.toDouble() / SAMPLE_RATE else 0.0
        if (audio == null || seconds < 0.25) {
            if (started != null) onDropped()
            return
        }
        deliveries.put(audio to if (started != null) now() - started else seconds)
        if (deliverer?.isAlive != true) {
            deliverer = Thread({ deliver() }, "utterances").apply {
                isDaemon = true
                start()
            }
        }
    }

    /** Utterances to the handler, one after the other. */
    private fun deliver() {
        while (true) {
            val (audio, seconds) = deliveries.take()
            try {
                onUtterance(audio, seconds)
            } catch (exc: Exception) { // a bad utterance must not stop listening
                VoiceLog.warn("utterance handler failed: ${VoiceLog.said(exc)}")
            }
        }
    }

    private fun concat(parts: List<FloatArray>): FloatArray {
        val out = FloatArray(parts.sumOf { it.size })
        var at = 0
        for (part in parts) {
            part.copyInto(out, at)
            at += part.size
        }
        return out
    }
}

/**
 * Speech to text with Parakeet, loaded once and kept. `stt.ParakeetSTT`: never raises out of
 * [transcribe], and a model that is not there or will not load leaves the coach deaf with the
 * reason.
 */
class ParakeetStt(private val context: Context, private val dir: java.io.File) : Recogniser {
    val name = "parakeet"
    /** NVIDIA's Parakeet TDT-CTC 110M on the phone; the desktop runs the 0.6B v3 (Tai, 2026-09-23). */
    val model = "parakeet-tdt_ctc-110m"

    @Volatile private var asr: Parakeet? = null
    @Volatile private var failed: String? = null
    private val lock = Any()

    /** Recognitions hold it shared; [close] alone, so a model is never freed under one. */
    private val using = java.util.concurrent.locks.ReentrantReadWriteLock()

    @Volatile override var loadSeconds: Double? = null
        private set

    override val loaded: Boolean get() = asr != null

    override fun healthcheck(): Pair<Boolean, String> {
        failed?.let { return false to "$model failed to load: $it" }
        val where = if (asr != null) "cpu" else "cpu (not loaded yet)"
        return true to "$model, $where"
    }

    override fun load(): Boolean {
        synchronized(lock) {
            if (asr != null) return true
            // A model not downloaded yet is looked for again: Setup fetches it after the first
            // warm-up, and the recogniser stayed off until a restart (voice finding 4). Any
            // other failure stands.
            if (failed != null && failed != NOT_DOWNLOADED) return false
            failed = null
            if (!Parakeet.FILES.all { java.io.File(dir, it).exists() }) {
                // The files come from Setup only, as the desktop's offline hub has it; the phone's
                // Setup is in the drawer, where the desktop's is under Help.
                failed = NOT_DOWNLOADED
                VoiceLog.warn("Parakeet unavailable, speech recognition is off: $failed")
                return false
            }
            return try {
                val began = SystemClock.elapsedRealtime()
                asr = Parakeet.load(context, dir, THREADS)
                loadSeconds = (SystemClock.elapsedRealtime() - began) / 1000.0
                true
            } catch (exc: Throwable) { // degrade to off, never crash the session
                failed = "${exc::class.java.simpleName}: ${exc.message}"
                VoiceLog.warn("Parakeet unavailable, speech recognition is off: ${VoiceLog.said(exc)}")
                false
            }
        }
    }

    /**
     * Half a second of silence through the model, so the first real clip does not pay the
     * runtime's warm-up. The seconds it took, null if the recogniser did not load.
     */
    override fun prime(): Double? {
        if (!load()) return null
        val began = SystemClock.elapsedRealtime()
        transcribe(FloatArray(SAMPLE_RATE / 2))
        return (SystemClock.elapsedRealtime() - began) / 1000.0
    }

    /** Raw text for a float32 mono clip at 16 kHz; empty when nothing was heard. */
    override fun transcribe(audio: FloatArray): String {
        if (!load()) return ""
        val shared = using.readLock()
        shared.lock()
        val text = try {
            val model = asr ?: return ""
            // A partial and the final result may run at once, as they do on the desktop: the
            // sessions are safe to share, and the decoding keeps its state on the stack.
            model.recognize(audio)
        } catch (exc: Exception) { // a bad clip must not kill the recogniser
            VoiceLog.warn("Parakeet failed on a clip: ${VoiceLog.said(exc)}")
            return ""
        } finally {
            shared.unlock()
        }
        return Py.strip(text)
    }

    /**
     * Free the model (a rebuilt recogniser replaces this one): the desktop leaves it to the
     * collector, but the phone's sessions hold 700 MB of native memory no collector sees.
     * Waits for a recognition in flight.
     */
    override fun close() {
        val alone = using.writeLock()
        alone.lock()
        try {
            synchronized(lock) {
                runCatching { asr?.close() }
                asr = null
                failed = "closed"
            }
        } finally {
            alone.unlock()
        }
    }

    companion object {
        /** The encoder's threads: the phone's big cores, as the voice's. */
        const val THREADS = 4

        /** Why the recogniser is off before Setup fetched it: looked for again at each load. */
        const val NOT_DOWNLOADED = "not downloaded yet (Setup, in the menu)"
    }
}
