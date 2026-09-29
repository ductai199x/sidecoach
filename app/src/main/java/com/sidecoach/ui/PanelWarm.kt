package com.sidecoach.ui

import android.content.Context
import android.util.Log
import com.sidecoach.analysis.Assessing
import com.sidecoach.analysis.EngineAssets
import com.sidecoach.analysis.OpeningBook
import com.sidecoach.chess.Board
import com.sidecoach.coach.CoachRunner
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.combine
import com.sidecoach.llm.modelOf
import com.sidecoach.llm.pricesFor
import com.sidecoach.llm.usageOf
import com.sidecoach.vision.BoardDetect
import com.sidecoach.vision.BoardTracker
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.GameWatcher
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.File
import java.math.BigDecimal
import java.math.RoundingMode
import java.net.URI
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.thread
import kotlin.concurrent.withLock

/**
 * Everything a session needs, loaded when the app opens: `ui/panel.py`'s `warm`, and the
 * stages it builds once and keeps.
 *
 * The desktop loads the embedding net, the voice and speech recognition, Stockfish and the
 * coach before any Start, so Start only starts the screen and the splash can show each one
 * land (docs/app-plan.md, "everything loads when the app opens"). The phone does the same.
 * The engine and the coach are built here and kept across Starts, rebuilt only when a save
 * changed what they were built from; a save that can reach them does so now
 * ([applySettings], `panel.apply_settings` for what the phone has).
 */
object PanelWarm {

    private const val TAG = "sidecoach.warm"

    /**
     * The warm-up's stages, in the order they finish, as the splash screen shows them: one
     * square of a rank each, the pieces landing left to right. `panel.WARM_STAGES`.
     */
    val STAGES: List<Pair<String, String>> = listOf(
        "pieces" to "Piece recognition",
        "voice" to "Voice",
        "stt" to "Speech recognition",
        "mic" to "Microphone",
        "vision" to "Board reading",
        "engine" to "Stockfish",
        "coach" to "Coach",
        "ready" to "Ready",
    )

    /** `panel.THRESHOLD_FIELDS`: the coach settings the annotator's thresholds come from. */
    private val THRESHOLD_FIELDS = setOf(
        "inaccuracy", "mistake", "blunder", "decisive_gap", "only_move_gap", "excellent", "decided",
    ).map { "coach.$it" }.toSet()

    private val lock = Any()
    private val warmLock = ReentrantLock()
    private val stages = LinkedHashMap<String, JSONObject>()
    private var begun = 0L

    /** `_warm_done`: set when a warm-up has run through; cleared while one runs. */
    private val doneLock = Object()
    private var doneFlag = false

    /** The net, the engine and the coach: built once and kept (`_assessing`, `_coaching`). */
    @Volatile var embedder: CellEmbedder? = null
        private set
    @Volatile var assessing: Assessing? = null
        private set
    @Volatile var coaching: CoachRunner? = null
        private set
    private var engineStale = false

    /**
     * What builds the engine's stage, finds the engine's network and holds the opening tables:
     * `cli._Assessing`, `config.find_stockfish` and the data folder's `openings`, which the
     * desktop's tests monkeypatch or point at a folder of their own; a test here sets these.
     */
    @Volatile internal var makeEngine: (Context, SettingsStore) -> Assessing = { context, store ->
        Assessing.fromSettings(context, store)
    }
    @Volatile internal var findEngine: (Context) -> File? = { EngineAssets.found(it) }
    @Volatile internal var openings: (Context) -> File = { File(it.filesDir, "openings") }

    /**
     * The stages a test starts from, as the desktop's tests start from a fresh `PanelEngine`;
     * the ones it replaced, to put back.
     */
    internal fun keepStages(engine: Assessing?, coach: CoachRunner?): Pair<Assessing?, CoachRunner?> =
        warmLock.withLock {
            val before = assessing to coaching
            assessing = engine
            coaching = coach
            engineStale = false
            PanelVoice.coach = coach?.coach
            before
        }

    /** Whether a warm-up has been started in this process; the app opens it once. */
    @Volatile private var started = false

    private lateinit var appContext: Context

    /**
     * Load everything a session needs before any Start, off the caller's thread; a Start
     * that comes first waits for it. Once per process: the activity opening again finds it
     * done, as the desktop's window reopened from the tray finds its sidecar warm.
     */
    fun warm(context: Context) {
        synchronized(lock) {
            appContext = context.applicationContext
            if (started) return
            started = true
        }
        thread(name = "warm", isDaemon = true) { runWarm() }
    }

    val done: Boolean get() = synchronized(doneLock) { doneFlag }

    private fun runWarm() {
        warmLock.withLock {
            synchronized(doneLock) { doneFlag = false }
            begun = System.nanoTime()
            synchronized(lock) {
                stages.clear()
                for ((name, label) in STAGES) {
                    stages[name] = JSONObject().put("name", name).put("label", label).put("state", "pending")
                }
            }
            try {
                // The voice built, not loaded (`panel._build_tts` at construction): the page
                // knows the voices at once, and the voice stage below loads the model.
                PanelVoice.build(appContext, SettingsStore(appContext))
                Captions.configure(appContext, SettingsStore(appContext).captions())
                Captions.style(SettingsStore(appContext).captionStyle())
                Captions.keep(SettingsStore(appContext).captionsKeep())
                PanelListen.build(appContext)
                stage("pieces", "loading")
                // OpenCV's native library first: the net's prime and the vision's pass both
                // need it (the desktop has cv2 the moment it imports the vision).
                GameWatcher.loaded
                val net = runCatching { CellEmbedder.load(appContext) }
                    .onFailure { Log.w(TAG, "the piece net did not load: $it") }
                    .getOrNull()
                embedder = net
                if (net != null) {
                    val primed = net.prime()
                    PanelEngine.emit(
                        "sys", "info",
                        "pieces: embedding net on ${net.deviceDetail}, primed in ${f1(primed)}s",
                    )
                    stage("pieces", "done", "on ${net.deviceDetail}")
                } else {
                    PanelEngine.emit("sys", "warn", "pieces: silhouette overlap (no embedding net)")
                    stage("pieces", "off", "no embedding net; silhouettes instead")
                }
                warmVoice()
                stage("vision", "loading")
                warmPipeline()
                stage("vision", "done")
                stage("engine", "loading")
                stage("coach", "loading")
                val (engine, coach) = prepareStages()
                if (!engine.on) stage("engine", "off", engine.detail)
                if (!coach.on) stage("coach", "off", coach.detail)
                warmEngines()
                if (engine.on) stage("engine", "done", engine.detail)
                warmModel()
                if (coach.on) stage("coach", "done", coach.detail)
                stage("ready", "done")
            } catch (exc: Throwable) {
                // The session builds what is missing. A Throwable, not only an Exception:
                // on Android an Error on this thread (a native library not loaded) would
                // take the whole app down with it.
                PanelEngine.emit("sys", "warn", "warm-up: ${PanelEngine.pyRepr(exc)}")
                synchronized(lock) {
                    for (one in stages.values) {
                        val state = one.optString("state")
                        if (state == "pending" || state == "loading") {
                            one.put("state", "off")
                            one.put("detail", PanelEngine.pyRepr(exc))
                        }
                    }
                }
            } finally {
                synchronized(doneLock) {
                    doneFlag = true
                    doneLock.notifyAll()
                }
            }
        }
    }

    /** One warm-up stage's state, with the seconds since the warm-up began. `_stage`. */
    private fun stage(name: String, state: String, detail: String = "") {
        synchronized(lock) {
            val one = stages[name] ?: return
            one.put("state", state)
            one.put("detail", detail)
            one.put("seconds", round1((System.nanoTime() - begun) / 1e9))
        }
    }

    /** What the splash screen polls: every stage and whether the warm-up is over. */
    fun status(): JSONObject {
        val list = JSONArray()
        synchronized(lock) { for (one in stages.values) list.put(JSONObject(one.toString())) }
        val seconds = if (begun != 0L) round1((System.nanoTime() - begun) / 1e9) else 0.0
        return JSONObject().put("done", done).put("seconds", seconds).put("stages", list)
    }

    /**
     * A Start waits for a warm-up under way, as `_run` does, and then has the stages it
     * built -- or builds them, when there was none. Blocks: call it off the main thread.
     */
    fun forSession(context: Context): Triple<CellEmbedder?, Assessing, CoachRunner> {
        synchronized(lock) { if (!::appContext.isInitialized) appContext = context.applicationContext }
        if (!done && warmLock.isLocked) {
            PanelEngine.set("status" to "loading the models")
            synchronized(doneLock) {
                val until = System.currentTimeMillis() + 120_000
                while (!doneFlag) {
                    val left = until - System.currentTimeMillis()
                    if (left <= 0) break
                    doneLock.wait(left)
                }
            }
        }
        warmLock.withLock {
            if (embedder == null) {
                embedder = runCatching { CellEmbedder.load(appContext) }.getOrNull()
                embedder?.let {
                    PanelEngine.emit("sys", "info", "pieces: embedding net on ${it.deviceDetail}")
                }
            }
            val (engine, coach) = prepareStages()
            return Triple(embedder, engine, coach)
        }
    }

    /**
     * Stockfish and the coach, built once and kept; rebuilt only when a save changed what
     * they were built from. `_prepare_stages`.
     */
    private fun prepareStages(): Pair<Assessing, CoachRunner> {
        // Built before there was an engine, and there is one now (Setup fetched the
        // network after the warm-up): built again, and not only at the next launch.
        val kept = assessing
        if (kept != null && !kept.on && findEngine(appContext) != null) engineStale = true
        if (engineStale) {
            // The coach too: it was built knowing whether there was an engine at all.
            runCatching { assessing?.close() }
            runCatching { coaching?.close() }
            assessing = null
            coaching = null
            engineStale = false
        }
        // The opening tables fetched after the engine was built (Setup, on a first run): the
        // kept annotator takes the book now rather than at the next launch.
        val annotator = assessing?.analyser?.annotator
        if (annotator != null && annotator.book == null) {
            annotator.book = OpeningBook.load(openings(appContext))
        }
        val store = SettingsStore(appContext)
        val engine = assessing ?: makeEngine(appContext, store).also {
            assessing = it
            PanelEngine.emit("sys", if (it.on) "info" else "warn", "engine: ${it.detail}")
        }
        val coach = coaching ?: CoachRunner(
            store,
            enabled = engine.on,
            engine = if (engine.on) Assessing.coachEngine(appContext, store) else null,
            computed = engine.computed,
        ).also {
            coaching = it
            // The voice takes a model reply sentence by sentence as it is written.
            it.coach?.streamSink = { trigger, ply -> PanelVoice.streamSink(trigger, ply) }
            PanelEngine.emit("sys", if (it.on) "info" else "warn", "coach: ${it.detail}")
        }
        PanelVoice.coach = coach.coach
        return engine to coach
    }

    /**
     * `_warm_voice`: both voice models loaded once, off the pipeline thread, each primed with
     * one inference so the first real one is not the slow one, and where they landed said;
     * then the microphone, and listening in the mode the settings name -- when there is speech
     * recognition to listen with.
     */
    private fun warmVoice() {
        val tts = PanelVoice.tts
        if (tts == null) {
            stage("voice", "off", "voice off")
        } else {
            stage("voice", "loading")
            val ok = tts.load()
            val detail = tts.healthcheck().second
            if (ok) {
                val primed = tts.prime()
                PanelEngine.emit(
                    "sys",
                    "info",
                    "voice: $detail; loaded once in ${f1(tts.loadSeconds ?: 0.0)}s" +
                        (if (primed != null) ", primed in ${f1(primed)}s" else ""),
                    JSONObject().put("device", "cpu"),
                )
                stage("voice", "done", detail)
            } else {
                PanelEngine.emit("sys", "warn", "voice: $detail")
                stage("voice", "off", detail)
            }
        }
        val stt = PanelListen.stt
        if (stt == null) {
            stage("stt", "off", "speech recognition off")
        } else {
            stage("stt", "loading")
            val ok = stt.load()
            val detail = stt.healthcheck().second
            if (ok) {
                val primed = stt.prime()
                PanelEngine.emit(
                    "sys",
                    "info",
                    "speech recognition: $detail; loaded once in ${f1(stt.loadSeconds ?: 0.0)}s" +
                        (if (primed != null) ", primed in ${f1(primed)}s" else ""),
                    JSONObject().put("device", "cpu"),
                )
                stage("stt", "done", detail)
            } else {
                PanelEngine.emit("sys", "warn", "speech recognition: $detail")
                stage("stt", "off", detail)
            }
        }
        PanelVoice.tts?.let { PanelEngine.set("voice_detail" to it.healthcheck().second) }
        val (sttReady, sttDetail) = stt?.healthcheck() ?: (false to "speech recognition off")
        PanelEngine.set("stt_detail" to sttDetail, "stt_ready" to sttReady)
        val (ok, detail) = PanelListen.mic?.healthcheck() ?: (false to "no microphone")
        PanelEngine.emit("sys", if (ok) "info" else "warn", "mic: $detail")
        stage("mic", "loading")
        val wanted = SettingsStore(appContext).listen()
        if (wanted != "off") {
            // Speech recognition that is off has no use for the microphone: it does not open.
            if (stt?.healthcheck()?.first == true) {
                PanelListen.setListen(wanted)
            } else {
                PanelEngine.emit("voice", "warn", "listening: off (speech recognition is not available)")
            }
        }
        stage("mic", if (ok) "done" else "off", detail)
    }

    /**
     * The vision's first pass paid (OpenCV sets itself up on the first board search), so the
     * first frame is read at speed. Nothing here touches the screen. `_warm_pipeline`.
     */
    private fun warmPipeline() {
        val began = System.nanoTime()
        if (!GameWatcher.loaded) return
        val blank = Mat.zeros(1080, 1920, CvType.CV_8UC3)
        try {
            BoardTracker().update(BoardDetect.toGray(blank))
        } finally {
            blank.release()
        }
        PanelEngine.emit("sys", "info", "vision: first pass primed in ${f1((System.nanoTime() - began) / 1e9)}s")
    }

    /**
     * One search of the start position, so the first real one does not pay the hash
     * allocation and the thread start. `_warm_engines`; the phone has no second engine
     * for named moves (see CoachRunner).
     */
    private fun warmEngines() {
        val analyser = assessing?.analyser ?: return
        val began = System.nanoTime()
        runCatching { analyser.annotator.engine.analyse(Board(), depth = analyser.annotator.depth) }
        PanelEngine.emit("sys", "info", "engine: primed in ${f1((System.nanoTime() - began) / 1e9)}s")
    }

    /**
     * A model server on this machine loads its model on the first request; one short request
     * now, so the coach's first line is not the one that waits. A paid provider is never
     * called for this. `_warm_model`.
     */
    private fun warmModel() {
        val provider = coaching?.coach?.speaker?.provider ?: return
        if (provider is OfflineProvider) return
        val base = (provider as? OpenAICompatProvider)?.baseUrl ?: return
        val host = runCatching { URI(base).host }.getOrNull() ?: ""
        if (host !in setOf("localhost", "127.0.0.1", "::1")) return
        val began = System.nanoTime()
        try {
            provider.chat(
                "Reply with the one word: ready.",
                listOf(ChatMessage("user", "Ready?")),
                maxTokens = 8,
                temperature = 0.0,
            )
        } catch (exc: Exception) {
            PanelEngine.emit("sys", "warn", "coach: the local model did not answer a warm-up: ${exc.message ?: ""}")
            return
        }
        PanelEngine.emit("sys", "info", "coach: local model answered in ${f1((System.nanoTime() - began) / 1e9)}s")
    }

    /**
     * The settings saved: what the kept voice, hearing, coach and engine can change, they do
     * now; the list of dotted names applied. Each change on its own: one that fails is named in
     * the log and the others still go. `panel.apply_settings`, less what the phone does not
     * have: a recording, a capture backend, an output device to choose and a push-to-talk key.
     */
    fun applySettings(store: SettingsStore, changed: List<String>): List<String> {
        val applied = ArrayList<String>()
        fun attempt(names: List<String>, apply: () -> Unit) {
            val hit = names.filter { it in changed }
            if (hit.isEmpty()) return
            try {
                apply()
            } catch (exc: Exception) {
                PanelEngine.emit("sys", "warn", "${hit.joinToString(", ")}: not applied (${exc.message ?: ""})")
                return
            }
            applied.addAll(hit)
        }
        if (changed.any { it == "engine.stockfish_path" || it == "tools_dir" }) engineStale = true
        attempt(listOf("voice.voice")) { PanelVoice.setVoice(store.voice()) }
        attempt(listOf("voice.listen")) { PanelListen.setListen(store.listen()) }
        attempt(listOf("voice.speed")) { PanelVoice.tts?.speed = store.speed() }
        attempt(listOf("voice.captions")) { Captions.configure(appContext, store.captions(), ask = true) }
        attempt(CAPTION_LOOK) { Captions.style(store.captionStyle()) }
        attempt(listOf("voice.captions_keep")) { Captions.keep(store.captionsKeep()) }
        // Saved, and seen at once: a sample line in the new look over the settings page.
        if (changed.any { it == "voice.captions" || it in CAPTION_LOOK }) Captions.preview()
        if (changed.any { it in VOICE_RELOAD }) {
            // A new language, folder or device: the old voice is cut off, the new one built
            // and loaded off the caller's thread. `_reload_voice`.
            runCatching { PanelVoice.tts?.stop() }
            PanelVoice.build(appContext, store)
            PanelEngine.emit("voice", "info", "voice reloading: ${PanelVoice.tts?.healthcheck()?.second}")
            thread(name = "voice-reload", isDaemon = true) { warmVoice() }
            applied.addAll(changed.filter { it in VOICE_RELOAD })
        }
        if (changed.any { it in STT_RELOAD }) {
            PanelListen.reload { warmVoice() }
            applied.addAll(changed.filter { it in STT_RELOAD })
        }
        for (name in changed) if (name.startsWith("llm.price_")) applied.add(name)
        val runner = coaching
        val coach = runner?.coach
        if (runner != null && coach != null) {
            val section = store.section("coach")
            attempt(listOf("coach.stuck_seconds")) {
                coach.policy.settings = coach.policy.settings.copy(
                    stuckSeconds = section.optDouble("stuck_seconds", coach.policy.settings.stuckSeconds),
                )
            }
            attempt(listOf("coach.quiet")) {
                coach.policy.settings = coach.policy.settings.copy(quiet = section.optBoolean("quiet", false))
            }
            attempt(listOf("coach.persona", "profile.rating_low", "profile.rating_high")) {
                coach.reword(store.persona(), store.rating())
            }
            attempt(listOf("coach.local_verdicts")) { coach.localVerdicts = store.localVerdicts() }
            attempt(listOf("coach.verbalized_sampling")) { coach.verbalizedSampling = store.verbalizedSampling() }
            attempt(listOf("coach.summarise_every")) { coach.summariseEvery = store.summariseEvery() }
            attempt(listOf("coach.compact_record")) { coach.conversation.compactRecord = store.compactRecord() }
            val modelFields = changed.filter {
                (it.startsWith("llm.") && !it.startsWith("llm.price_")) ||
                    it == "coach.verify" || it == "coach.verify_model"
            }
            attempt(modelFields) {
                runner.reprovision(store)
                PanelEngine.emit("sys", "info", "coach: ${runner.detail}")
            }
        }
        val analyser = assessing?.analyser
        if (analyser != null) {
            val config = store.engine()
            val engine = analyser.annotator.engine
            attempt(changed.filter { it in THRESHOLD_FIELDS }) {
                analyser.annotator.thresholds = store.thresholds()
            }
            attempt(listOf("engine.depth")) { analyser.annotator.depth = config.depth }
            attempt(listOf("engine.multipv")) {
                analyser.annotator.multipv = config.multipv
                engine.multipv = config.multipv
            }
            attempt(listOf("engine.pv_plies")) { engine.pvPlies = config.pvPlies }
            attempt(listOf("engine.score_model")) { engine.scoreModel = config.scoreModel }
            attempt(listOf("engine.max_seconds")) { engine.maxSeconds = config.maxSeconds }
            attempt(listOf("engine.threads", "engine.hash_mb")) {
                engine.reconfigure(threads = config.threads, hashMb = config.hashMb)
            }
        }
        return applied.toSortedSet().toList()
    }

    /**
     * The session's model calls for the status strip: the game coach's ledger and the puzzle
     * rooms', combined -- the offline coach's too, which has one with nothing on it; null when
     * there is no coach. `_health`'s `llm`, asked once a second with the rest of the strip.
     */
    fun usage(runner: CoachRunner?): Map<String, Any?>? {
        val provider = runner?.coach?.speaker?.provider
        val usage = usageOf(provider) ?: return null
        val prices = pricesFor(SettingsStore(appContext).llm(), modelOf(provider))
        val ledgers = ArrayList<Map<String, Any?>>()
        ledgers.add(usage.summary(prices))
        runCatching { for (room in PuzzleRooms.sharedOrNull()?.usages().orEmpty()) ledgers.add(room.summary(prices)) }
        return combine(ledgers)
    }

    /** `panel.VOICE_RELOAD`: the voice settings that need a new voice, not a changed one. */
    private val VOICE_RELOAD = setOf("voice.lang", "voice.model_dir", "voice.tts_device")

    /** `panel.STT_RELOAD`: the settings that rebuild speech recognition and the microphone. */
    private val STT_RELOAD = setOf("voice.stt_device", "voice.mic_device")

    /** The captions' look (the phone's own settings): applied to the box at once. */
    private val CAPTION_LOOK = listOf(
        "voice.captions_text_color",
        "voice.captions_background",
        "voice.captions_opacity",
        "voice.captions_text_size",
    )

    /** Python's `f"{x:.1f}"`: half to even. */
    private fun f1(value: Double): String =
        BigDecimal(value).setScale(1, RoundingMode.HALF_EVEN).toPlainString()

    /** Python's `round(x, 1)`. */
    private fun round1(value: Double): Double =
        BigDecimal(value).setScale(1, RoundingMode.HALF_EVEN).toDouble()
}
