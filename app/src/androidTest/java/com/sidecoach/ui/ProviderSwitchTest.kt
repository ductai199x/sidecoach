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
            val custom = JSONObject()
                .put("provider", "custom")
                .put("base_url", "http://192.168.1.20:1234/v1")
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
            assertEquals("http://192.168.1.20:1234/v1", llm.getString("base_url"))
        } finally {
            store.save(saved)
        }
    }
}
