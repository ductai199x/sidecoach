package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Computed
import com.sidecoach.analysis.Engine
import com.sidecoach.chess.Board
import com.sidecoach.coach.CoachRunner
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Base64

/**
 * The settings page as the phone serves it (Tai, 2026-09-24: "when I go to settings, and then
 * Voice, I only see American Female, Sarah ... check all other options thoroughly too"): the
 * voices the phone has, no field the phone does not honour, and the two that were cut off on
 * the way in -- the frame rate and a service's extra request fields -- reaching the code.
 */
@RunWith(AndroidJUnit4::class)
class SettingsPhoneTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun fields(described: JSONObject): Map<String, List<String>> {
        val schema = described.getJSONObject("schema")
        val props = schema.getJSONObject("properties")
        val defs = schema.getJSONObject("\$defs")
        return props.keys().asSequence().associateWith { name ->
            val ref = props.getJSONObject(name).optString("\$ref")
            if (ref.isEmpty()) emptyList()
            else defs.getJSONObject(ref.substringAfterLast('/')).getJSONObject("properties").keys().asSequence().toList()
        }
    }

    @Test
    fun theVoiceSettingOffersEveryEnglishVoice() {
        val voices = SettingsApi(SettingsStore(app)).describe().getJSONArray("voices")
        val names = (0 until voices.length()).map { voices.getString(it) }
        assertTrue("only $names", names.size > 10)
        assertTrue(names.toString(), "af_sarah" in names && "bm_george" in names)
    }

    @Test
    fun noFieldThePhoneDoesNotHonourIsOffered() {
        val got = fields(SettingsApi(SettingsStore(app)).describe())
        for (gone in listOf("gpu", "data_dir", "tools_dir")) assertFalse(gone, gone in got)
        assertEquals(listOf("fps"), got["capture"])
        for (gone in listOf("model_dir", "tts_device", "stt_device", "mic_device", "output_device", "ptt_key")) {
            assertFalse("voice.$gone", gone in got.getValue("voice"))
        }
        assertFalse("engine.stockfish_path" in got.getValue("engine").map { "engine.$it" })
        assertTrue("voice.captions_keep", "captions_keep" in got.getValue("voice"))
        // No development field either (config.py's `dev`): the page is never the desktop's shell.
        for (gone in listOf("api_key_env", "thinking_control", "extra_body")) assertFalse("llm.$gone", gone in got.getValue("llm"))
    }

    /**
     * What the page says is true of the phone (Tai: plain language, true for the platform): no
     * push-to-talk key held from any window, no model on "this computer", the engine's phone
     * defaults -- and the marks the page reads (`test_the_schema_carries_the_marks_the_page_reads`).
     */
    @Test
    fun thePageDescribesThePhone() {
        val described = SettingsApi(SettingsStore(app)).describe()
        val keys = described.keys().asSequence().toSet()
        assertTrue(keys.toString(), keys.containsAll(setOf("settings", "schema", "path", "secrets", "keyring", "devices")))
        val defs = described.getJSONObject("schema").getJSONObject("\$defs")
        val listen = defs.getJSONObject("VoiceSettings").getJSONObject("properties").getJSONObject("listen")
        assertEquals(listOf("off", "ptt", "vad"), listen.getJSONArray("choices").let { a -> (0 until a.length()).map { a.getString(it) } })
        assertEquals("now", listen.getString("applies"))
        assertEquals("Listening" to "Hearing you", listen.getString("label") to listen.getString("section"))
        assertTrue(listen.getString("description").startsWith("How the coach hears you"))
        assertTrue(listen.getJSONObject("choice_labels").getString("ptt").startsWith("While I hold"))
        val engine = defs.getJSONObject("EngineSettings").getJSONObject("properties")
        assertEquals(2 to 64, engine.getJSONObject("threads").getInt("default") to engine.getJSONObject("hash_mb").getInt("default"))
        val llm = defs.getJSONObject("LLMSettings").getJSONObject("properties")
        val words = listOf(listen, engine.getJSONObject("hash_mb"), llm.getJSONObject("provider"), llm.getJSONObject("base_url")).joinToString(" ")
        for (desktop in listOf("from any window", "push-to-talk key", "this computer", "256 is plenty")) {
            assertFalse("\"$desktop\" is the desktop's", desktop in words)
        }
        assertEquals(45, described.getJSONObject("settings").getJSONObject("coach").getInt("stuck_seconds"))
        assertTrue("OPENAI_API_KEY" in described.getJSONObject("secrets").keys().asSequence().toList())
    }

    /**
     * A save says what applies now and what at the next Start; one that does not fit is refused
     * by field and nothing is saved; a key is stored apart and taken away again
     * (`test_a_save_writes_the_file_and_says_what_applies_when`, the secrets' round trip).
     */
    @Test
    fun aSaveSaysWhatAppliesWhenAndRefusesWhatDoesNotFit() {
        val store = SettingsStore(app)
        val api = SettingsApi(store)
        val saved = store.values()
        try {
            val values = store.values()
            values.getJSONObject("coach").put("stuck_seconds", 60)
            values.getJSONObject("voice").put("voice", "am_adam")
            values.getJSONObject("capture").put("fps", 6)
            val res = api.update(JSONObject().put("settings", values))
            assertTrue(res.toString(), res.getBoolean("ok"))
            assertEquals("[\"coach.stuck_seconds\",\"voice.voice\"]", res.getJSONArray("now").toString())
            assertEquals("[\"capture.fps\"]", res.getJSONArray("next").toString())
            assertEquals(0, res.getJSONArray("restart").length())
            assertEquals(60, store.section("coach").getInt("stuck_seconds"))
            val before = store.values().toString()
            values.getJSONObject("engine").put("depth", 999)
            val refused = api.update(JSONObject().put("settings", values))
            assertFalse(refused.getBoolean("ok"))
            assertEquals("[\"engine\",\"depth\"]", refused.getJSONArray("errors").getJSONObject(0).getJSONArray("loc").toString())
            assertEquals("nothing was saved", before, store.values().toString())
            values.getJSONObject("engine").put("depth", 18)
            val keyed = api.update(JSONObject().put("settings", values).put("secrets", JSONObject().put("OPENAI_API_KEY", "sk-test")))
            assertTrue(keyed.getJSONObject("secrets").getBoolean("OPENAI_API_KEY") && store.secretSet("OPENAI_API_KEY"))
            assertFalse("the key is not in the settings", "sk-test" in store.values().toString())
            api.update(JSONObject().put("settings", values).put("secrets", JSONObject().put("OPENAI_API_KEY", JSONObject.NULL)))
            assertFalse(store.secretSet("OPENAI_API_KEY"))
        } finally {
            store.setSecret("OPENAI_API_KEY", null)
            api.update(JSONObject().put("settings", saved))
        }
    }

    /**
     * The page sends what changed, and only that: a save of one field keeps every other as it
     * is (Tai's phone, 2026-10-01: a save of the captions reset all of his settings to the
     * defaults, and each caption switch saved alone turned the other off). The desktop's
     * `test_a_save_changes_only_what_it_sends`.
     */
    /**
     * What the panel's page sends since 2026-10-01, over the phone's own routes (the shared page
     * has two servers: every new request shape is served here too). Muted and the listening mode
     * are kept in the settings for the next session. The header's CC and pin save the captions
     * alone, and the poll says so, which is what the buttons follow. A player's own sets are
     * made, given more images, renamed, deleted and put back.
     */
    @Test
    fun thePagesNewRequestsAreServedAndKept() {
        val store = SettingsStore(app)
        val saved = store.values()
        val api = PanelApi(
            "test",
            SetupApi.shared(app),
            SettingsApi(store),
            files = app.filesDir,
            rooms = { PuzzleRooms.shared(app) },
            dumps = File(app.cacheDir, "dumps"),
        )
        fun call(method: String, path: String, body: JSONObject = JSONObject()): JSONObject {
            val reply = api.handle(method, path, emptyMap(), body.toString(), ByteArray(0))
            assertNotNull("$method $path is not served", reply)
            assertEquals("$method $path", 200, reply!!.status)
            return JSONObject(String(reply.body))
        }
        fun captions(): JSONObject = PanelEngine.poll(0).getJSONObject("state").getJSONObject("captions")
        val png = Base64.getEncoder().encodeToString(
            byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 13, 10, 26, 10) + ByteArray(16),
        )
        fun image(name: String) = JSONArray().put(JSONObject().put("name", name).put("data", png))
        var made: String? = null
        try {
            assertTrue(call("POST", "/api/muted", JSONObject().put("muted", true)).getBoolean("ok"))
            assertTrue("muted is kept", SettingsStore(app).muted())
            assertTrue(PanelVoice.muted)
            call("POST", "/api/muted", JSONObject().put("muted", false))
            assertFalse(SettingsStore(app).muted())
            assertEquals("off", call("POST", "/api/listen", JSONObject().put("mode", "off")).getString("mode"))
            assertEquals("the listening mode is kept", "off", SettingsStore(app).listen())

            val both = JSONObject().put("captions", true).put("captions_keep", true)
            assertTrue(call("PUT", "/api/settings", JSONObject().put("settings", JSONObject().put("voice", both)).put("secrets", JSONObject())).getBoolean("ok"))
            assertTrue(captions().getBoolean("on") && captions().getBoolean("keep"))
            val off = JSONObject().put("captions", false)
            assertTrue(call("PUT", "/api/settings", JSONObject().put("settings", JSONObject().put("voice", off)).put("secrets", JSONObject())).getBoolean("ok"))
            assertFalse("CC follows the captions turned off", captions().getBoolean("on"))
            assertTrue("the pin's setting is kept", captions().getBoolean("keep"))
            assertEquals(saved.getJSONObject("profile").toString(), store.values().getJSONObject("profile").toString())

            val first = call("POST", "/api/pieces/own", JSONObject().put("name", "Phone test").put("files", image("wK.png")))
            assertTrue(first.toString(), first.getBoolean("ok"))
            made = first.getString("id").removePrefix(Looks.OWN)
            val id = made
            assertEquals("Phone wood", call("PUT", "/api/pieces/own/$id", JSONObject().put("name", "Phone wood")).getString("name"))
            assertEquals("bQ", call("POST", "/api/pieces/own/$id/images", JSONObject().put("files", image("bQ.png"))).getJSONArray("taken").getString(0))
            fun labels(): List<String> {
                val own = Looks.catalogue(store.files).getJSONArray("own")
                return (0 until own.length()).map { own.getJSONObject(it).getString("label") }
            }
            assertTrue(labels().contains("Phone wood"))
            assertTrue(call("DELETE", "/api/pieces/own/$id").getBoolean("ok"))
            assertFalse(labels().contains("Phone wood"))
            assertTrue(call("POST", "/api/pieces/own/$id/restore").getBoolean("ok"))
            assertTrue(labels().contains("Phone wood"))
            assertFalse("no such set", call("PUT", "/api/pieces/own/nope", JSONObject().put("name", "x")).getBoolean("ok"))
            // The settings' counter the page reads with each poll: raised by a save and by a
            // change to a set, not by a refusal.
            fun version(): Int = PanelEngine.poll(0).getJSONObject("state").getInt("settings_version")
            val before = version()
            val pick = JSONObject().put("board", JSONObject().put("pieces", Looks.OWN + id))
            assertTrue(call("PUT", "/api/settings", JSONObject().put("settings", pick).put("secrets", JSONObject())).getBoolean("ok"))
            assertTrue("a save raises it", version() > before)
            val afterSave = version()
            call("PUT", "/api/pieces/own/$id", JSONObject().put("name", "Phone oak"))
            assertTrue("a set renamed raises it", version() > afterSave)
            val afterRename = version()
            call("PUT", "/api/pieces/own/nope", JSONObject().put("name", "x"))
            assertEquals("nothing changed, nothing raised", afterRename, version())
        } finally {
            store.save(saved)
            made?.let {
                File(Looks.ownRoot(store.files), it).deleteRecursively()
                File(File(Looks.ownRoot(store.files), Looks.REMOVED), it).deleteRecursively()
            }
        }
    }

    @Test
    fun aSaveOfOneFieldKeepsEveryOther() {
        val store = SettingsStore(app)
        val api = SettingsApi(store)
        val saved = store.values()
        try {
            val mine = store.values()
            mine.getJSONObject("profile").put("chesscom_user", "nosegazm")
            mine.getJSONObject("board").put("pieces", "cburnett")
            mine.getJSONObject("voice").put("voice", "bm_george")
            mine.getJSONObject("voice").put("captions", false)
            mine.getJSONObject("voice").put("captions_keep", false)
            assertTrue(api.update(JSONObject().put("settings", mine)).getBoolean("ok"))

            val one = api.update(JSONObject().put("settings", JSONObject().put("voice", JSONObject().put("captions", true))))
            assertTrue(one.toString(), one.getBoolean("ok"))
            val two = api.update(JSONObject().put("settings", JSONObject().put("voice", JSONObject().put("captions_keep", true))))
            assertTrue(two.toString(), two.getBoolean("ok"))

            val now = store.values()
            assertEquals("nosegazm", now.getJSONObject("profile").getString("chesscom_user"))
            assertEquals("cburnett", now.getJSONObject("board").getString("pieces"))
            assertEquals("bm_george", now.getJSONObject("voice").getString("voice"))
            assertTrue("the first switch was turned off by the second's save", now.getJSONObject("voice").getBoolean("captions"))
            assertTrue(now.getJSONObject("voice").getBoolean("captions_keep"))
            // The rest of a section a save touches stays too, every section it does not touch.
            assertEquals(saved.getJSONObject("coach").toString(), now.getJSONObject("coach").toString())
            assertEquals(saved.getJSONObject("engine").toString(), now.getJSONObject("engine").toString())
        } finally {
            store.save(saved)
        }
    }

    /**
     * With a coach and an engine built, a save changes them in place: the persona and the rating
     * band in the system message, the nudge time, the thresholds, the engine's depth and lines
     * (`test_a_save_reaches_the_running_coach_and_engine`).
     */
    @Test
    fun aSaveReachesTheRunningCoachAndEngine() {
        val store = SettingsStore(app)
        val api = SettingsApi(store)
        val (_, engine, runner) = PanelWarm.forSession(app)
        val coach = runner.coach
        val analyser = engine.analyser
        org.junit.Assume.assumeTrue("no engine here to reach", coach != null && analyser != null)
        val saved = store.values()
        try {
            val values = store.values()
            values.getJSONObject("coach").put("persona", "playful").put("stuck_seconds", 30).put("blunder", 0.4)
            values.getJSONObject("engine").put("depth", 12).put("multipv", 3)
            values.getJSONObject("profile").put("rating_low", 1500).put("rating_high", 1700)
            val res = api.update(JSONObject().put("settings", values))
            assertTrue(res.toString(), res.getBoolean("ok") && res.getJSONArray("next").length() == 0)
            val now = res.getJSONArray("now").let { a -> (0 until a.length()).map { a.getString(it) }.toSet() }
            assertEquals(
                setOf("coach.persona", "coach.stuck_seconds", "coach.blunder", "engine.depth", "engine.multipv", "profile.rating_low", "profile.rating_high"),
                now,
            )
            assertTrue(coach!!.system, "1500 to 1700" in coach.system)
            assertEquals(30.0, coach.policy.settings.stuckSeconds, 0.0)
            assertEquals(0.4, analyser!!.annotator.thresholds.blunder, 0.0)
            assertEquals(12 to 3, analyser.annotator.depth to analyser.annotator.multipv)
            assertEquals(3, analyser.annotator.engine.multipv)
        } finally {
            api.update(JSONObject().put("settings", saved))
        }
    }

    /**
     * The verifier is the provider `coach.verify` names with the model `coach.verify_model`
     * names -- nothing of the coach's own model, address or effort (`cli._verifier`); "same" is
     * the coach's own model, and empty is none.
     */
    @Test
    fun theVerifierIsTheModelItsSettingsName() {
        val store = SettingsStore(app)
        val saved = store.values()
        try {
            val values = store.values()
            values.getJSONObject("coach").put("verify", " DashScope ").put("verify_model", "qwen3.8-flash")
            values.getJSONObject("llm").put("provider", "local").put("model", "coach-model").put("base_url", "http://192.168.1.20:1234/v1")
            store.save(values)
            store.setSecret("DASHSCOPE_API_KEY", "sk-test")
            val runner = CoachRunner(store, enabled = false)
            val own = com.sidecoach.llm.OfflineProvider()
            val verifier = runner.verifier(store, own) as com.sidecoach.llm.OpenAICompatProvider
            assertEquals("dashscope", verifier.name)
            assertEquals("qwen3.8-flash", verifier.model)
            assertEquals("https://dashscope-us.aliyuncs.com/compatible-mode/v1", verifier.baseUrl)
            assertEquals(60.0, verifier.timeoutS, 0.0)
            store.save(values.also { it.getJSONObject("coach").put("verify", "same") })
            assertTrue(runner.verifier(store, own) === own)
            store.save(values.also { it.getJSONObject("coach").put("verify", "") })
            assertEquals(null, runner.verifier(store, own))
        } finally {
            store.setSecret("DASHSCOPE_API_KEY", null)
            store.save(saved)
        }
    }

    /**
     * The coach's engine answers from what the analyser computed and starts its Stockfish only at
     * a miss (Tai, 2026-09-29); the coach closed, it is closed too, and searches no more
     * (`test_cli_parts`, desktop ce06c4e). The binary is not there: a runner that started it
     * would fail, and so would a search after the close.
     */
    @Test
    fun theCoachsEngineReadsTheAnalysersStoreAndClosesWithTheCoach() {
        val store = SettingsStore(app)
        val saved = store.values()
        try {
            val values = store.values()
            values.getJSONObject("llm").put("provider", "none")
            store.save(values)
            val computed = Computed()
            val runner = CoachRunner(store, enabled = true, engine = Engine(File(app.filesDir, "no-stockfish"), null), computed = computed)
            val engine = runner.coachEngine!!
            assertTrue("the analyser's store", engine.computed === computed)
            assertEquals("no Stockfish started before a search", 0, engine.searches)
            runner.close()
            assertTrue(engine.closed)
            assertEquals(emptyList<Line>(), engine.analyse(Board(), 14))
            assertEquals(0, engine.searches)
        } finally {
            store.save(saved)
        }
    }

    /**
     * A Settings save and a mute at the same moment: each read the settings, laid its change over
     * them and saved, and the later save put back what the earlier one changed (the settings
     * review, 2026-10-01; `test_two_writes_at_once_keep_both_changes`). One write at a time, the
     * whole read-merge-save: while the settings are held, a mute and a save both wait, and then
     * both changes stay.
     */
    @Test
    fun twoWritesAtOnceKeepBothChanges() {
        val store = SettingsStore(app)
        val api = SettingsApi(store)
        val saved = store.values()
        val wasMuted = PanelVoice.muted
        try {
            val start = store.values()
            start.getJSONObject("voice").put("muted", false)
            start.getJSONObject("profile").put("chesscom_user", "")
            store.save(start)
            val user = JSONObject().put("settings", JSONObject().put("profile", JSONObject().put("chesscom_user", "nosegazm")))
            val mute = Thread { PanelVoice.mute(api, true) }
            val save = Thread { api.update(user) }
            synchronized(SettingsApi.SAVING) {
                mute.start()
                mute.join(200)
                assertTrue("the mute waits for the settings", mute.isAlive)
                save.start()
                save.join(200)
                assertTrue("so does a save", save.isAlive)
            }
            mute.join(5000)
            save.join(5000)
            val now = store.values()
            assertTrue("the mute is kept", now.getJSONObject("voice").getBoolean("muted"))
            assertEquals("nosegazm", now.getJSONObject("profile").getString("chesscom_user"))
        } finally {
            PanelVoice.setMuted(wasMuted)
            store.save(saved)
        }
    }

    /**
     * A key saved alone reaches the running coach and Game Review's model at once, and a key
     * forgotten leaves both: the save changed no setting, so neither was made again until a
     * restart (the settings review, 2026-10-01; `test_a_key_pasted_alone_reaches_the_coach_and_the_review`).
     * OpenAI's preset at an address nothing answers: the model is made with the key and found
     * unreachable at once, and nothing is sent anywhere.
     */
    @Test
    fun aKeyPastedAloneReachesTheCoachAndTheReview() {
        val store = SettingsStore(app)
        val api = SettingsApi(store)
        val saved = store.values()
        val hadKey = store.secret("OPENAI_API_KEY")
        var runner: CoachRunner? = null
        var before: Pair<com.sidecoach.analysis.Assessing?, CoachRunner?>? = null
        fun key(value: String?): JSONObject =
            JSONObject().put("settings", JSONObject()).put("secrets", JSONObject().put("OPENAI_API_KEY", value ?: JSONObject.NULL))
        try {
            store.setSecret("OPENAI_API_KEY", null)
            val values = store.values()
            values.getJSONObject("llm").put("provider", "openai").put("base_url", "http://127.0.0.1:9/v1")
            store.save(values)
            val made = CoachRunner(store, enabled = true)
            runner = made
            before = PanelWarm.keepStages(null, made)
            assertTrue(made.detail, "is not set" in made.detail)
            val rooms = ReviewRooms(store, storePath = null, files = app.cacheDir, makeEngine = ReviewRoom.engineMaker(app, store))
            assertEquals(null, rooms.model().first)

            val res = api.update(key("sk-test"))
            assertTrue(res.toString(), res.getBoolean("ok"))
            assertFalse("the coach has the key now: ${made.detail}", "is not set" in made.detail)
            val first = rooms.model().first
            assertNotNull("Game Review has it too", first)
            api.update(key("sk-other"))
            assertTrue("a key fixed, a model made with it", rooms.model().first !== first)

            api.update(key(null))
            assertTrue("forgotten: ${made.detail}", "is not set" in made.detail)
            assertEquals(null, rooms.model().first)
        } finally {
            before?.let { PanelWarm.keepStages(it.first, it.second) }
            runner?.close()
            store.setSecret("OPENAI_API_KEY", hadKey)
            store.save(saved)
        }
    }

    /**
     * Saved settings the phone cannot read are kept in its files, once, and the log says so --
     * never run over by the defaults in silence (`test_a_settings_file_that_cannot_be_read_is_kept_not_run_over`).
     */
    @Test
    fun savedSettingsThatCannotBeReadAreKeptNotRunOver() {
        val store = SettingsStore(app)
        val prefs = app.getSharedPreferences("settings", android.content.Context.MODE_PRIVATE)
        val raw = prefs.getString("json", null)
        val torn = "{\"profile\": {\"chesscom_user\": \"nosegazm-${System.nanoTime()}" // cut short
        fun copies(): List<File> =
            app.filesDir.listFiles { f -> f.name.startsWith("settings.unreadable-") }.orEmpty().filter { it.readText() == torn }
        try {
            prefs.edit().putString("json", torn).commit()
            val cursor = PanelEngine.poll(0).getInt("cursor")
            assertEquals("the defaults meanwhile", "", store.values().getJSONObject("profile").optString("chesscom_user"))
            assertEquals(1, copies().size)
            store.values()
            assertEquals("one copy a text", 1, copies().size)
            val lines = PanelEngine.poll(cursor).getJSONArray("lines")
            val name = copies()[0].name
            assertTrue("the log names the copy", (0 until lines.length()).any { name in lines.getJSONObject(it).getString("text") })
        } finally {
            copies().forEach { it.delete() }
            if (raw == null) prefs.edit().remove("json").commit() else prefs.edit().putString("json", raw).commit()
        }
    }

    @Test
    fun theFrameRateAndExtraFieldsReachTheCode() {
        val store = SettingsStore(app)
        val saved = store.values()
        try {
            val values = store.values()
            values.getJSONObject("capture").put("fps", 6)
            values.getJSONObject("llm").put(
                "extra_body",
                JSONObject("""{"enable_thinking": false, "extra": {"n": 2, "tags": ["a", "b"]}}"""),
            )
            store.save(values)
            assertEquals(6, store.captureFps())
            val extra = store.llm().extraBody
            assertEquals(false, extra["enable_thinking"])
            assertEquals(mapOf("n" to 2, "tags" to listOf("a", "b")), extra["extra"])
            values.getJSONObject("capture").put("fps", 99)
            store.save(values)
            assertEquals("kept in range", 30, store.captureFps())
        } finally {
            store.save(saved)
        }
    }
}
