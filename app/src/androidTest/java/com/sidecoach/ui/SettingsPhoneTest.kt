package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.Computed
import com.sidecoach.analysis.Engine
import com.sidecoach.chess.Board
import com.sidecoach.coach.CoachRunner
import com.sidecoach.events.Line
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

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
