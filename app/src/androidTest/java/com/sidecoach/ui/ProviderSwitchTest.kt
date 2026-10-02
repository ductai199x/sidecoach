package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.llm.KeyStore
import com.sidecoach.llm.LLMSettings
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.makeProvider
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * A switch of AI service leaves nothing of the old one: a hosted preset's address and key are
 * its own, and a save that changes the service without naming a model clears the model. The
 * desktop's `test_a_switch_of_service_leaves_nothing_of_the_old_one` (test_llm_factory.py) and
 * `test_a_new_service_does_not_take_the_old_ones_model` (test_settings_api.py).
 */
@RunWith(AndroidJUnit4::class)
class ProviderSwitchTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun aSwitchOfServiceLeavesNothingOfTheOldOne() {
        val keys = KeyStore { name -> if (name == "OPENAI_API_KEY") "sk-openai" else null }
        // What a custom service left in the settings: its address, its key variable.
        val left = LLMSettings(
            provider = "openai",
            baseUrl = "http://192.168.1.20:1234/v1",
            apiKeyEnv = "CUSTOM_API_KEY",
        )
        val provider = makeProvider(left, keys = keys) as OpenAICompatProvider
        assertEquals("https://api.openai.com/v1", provider.baseUrl)
        // The local server keeps its own address.
        val local = makeProvider(
            LLMSettings(provider = "local", baseUrl = "http://192.168.1.20:1234/v1"),
            keys = keys,
        ) as OpenAICompatProvider
        assertEquals("http://192.168.1.20:1234/v1", local.baseUrl)
    }

    @Test
    fun aNewServiceDoesNotTakeTheOldOnesModel() {
        val store = SettingsStore(app)
        val api = SettingsApi(store)
        val saved = store.values()
        try {
            // A server address that answers at once (nothing listens on the loopback's port 9):
            // each save rebuilds the kept coach's model, which asks the server whether it
            // answers, and a LAN address that is not there held each save until the attempt gave
            // up -- 21 s of this test on the emulator, for an answer it does not look at.
            val custom = JSONObject()
                .put("provider", "custom")
                .put("base_url", SERVER)
                .put("model", "qwen3")
            assertTrue(api.update(JSONObject().put("settings", JSONObject().put("llm", custom))).getBoolean("ok"))
            // The page sends what changed: the service alone.
            val openai = JSONObject().put("llm", JSONObject().put("provider", "openai"))
            assertTrue(api.update(JSONObject().put("settings", openai)).getBoolean("ok"))
            assertTrue(store.values().getJSONObject("llm").isNull("model"))
            // A model sent with the service is kept, and the local address with it.
            val local = JSONObject().put("llm", JSONObject().put("provider", "local").put("model", "mine"))
            assertTrue(api.update(JSONObject().put("settings", local)).getBoolean("ok"))
            val llm = store.values().getJSONObject("llm")
            assertEquals("mine", llm.getString("model"))
            assertEquals(SERVER, llm.getString("base_url"))
        } finally {
            store.save(saved)
        }
    }

    /**
     * Tai, 2026-10-02: openai (gpt-5.6-luna) -> local -> openai came back on OpenAI's default;
     * switching service and back restores that service's model. The desktop's
     * `test_each_service_keeps_its_own_model`.
     */
    @Test
    fun eachServiceKeepsItsOwnModel() {
        val store = SettingsStore(app)
        val api = SettingsApi(store)
        val saved = store.values()
        fun save(llm: JSONObject) =
            assertTrue(api.update(JSONObject().put("settings", JSONObject().put("llm", llm))).getBoolean("ok"))
        fun llm() = store.values().getJSONObject("llm")
        try {
            // From the default AI settings, as the desktop's test starts from a fresh settings
            // file: the app's own store keeps the models a test before this one left (the
            // service switches of aNewServiceDoesNotTakeTheOldOnesModel), and the set asserted
            // below is of this test's switches alone.
            store.save(JSONObject(saved.toString()).put("llm", store.defaults().getJSONObject("llm")))
            save(JSONObject().put("provider", "openai").put("model", "gpt-5.6-luna"))
            save(JSONObject().put("provider", "local")) // the page sends the service alone
            assertTrue(llm().isNull("model")) // never seen: its default
            save(JSONObject().put("model", "qwen3.8-35b"))
            save(JSONObject().put("provider", "openai"))
            assertEquals("gpt-5.6-luna", llm().getString("model"))
            save(JSONObject().put("provider", "local"))
            assertEquals("qwen3.8-35b", llm().getString("model"))
            // A service left on its default comes back on its default.
            save(JSONObject().put("provider", "dashscope"))
            assertTrue(llm().isNull("model"))
            save(JSONObject().put("provider", "local"))
            save(JSONObject().put("provider", "dashscope"))
            assertTrue(llm().isNull("model"))
            val models = llm().getJSONObject("models")
            assertEquals(setOf("openai", "local"), models.keys().asSequence().toSet())
            assertEquals("gpt-5.6-luna", models.getString("openai"))
            assertEquals("qwen3.8-35b", models.getString("local"))
            // Never a field on the page.
            val schema = api.describe().getJSONObject("schema")
            val field = schema.getJSONObject("\$defs").getJSONObject("LLMSettings")
                .getJSONObject("properties").getJSONObject("models")
            assertTrue(field.getBoolean("kept"))
        } finally {
            store.save(saved)
        }
    }

    private companion object {
        const val SERVER = "http://127.0.0.1:9/v1"
    }
}
