package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

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
