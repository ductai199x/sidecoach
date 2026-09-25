package com.sidecoach.ui

import com.sidecoach.voice.Voices
import org.json.JSONArray
import org.json.JSONObject

/**
 * The settings over HTTP, for the panel's settings page. `sidecoach/ui/settings_api.py`.
 *
 * GET gives the settings, the JSON schema the page renders from (every field's description,
 * its choices, and whether a change applies now, at the next Start or after a restart),
 * where they live, which keys are set, and the personas. PUT saves them, stores the keys in
 * the store of their own, and answers with what took effect now, what waits for the next
 * Start and what needs a restart.
 *
 * The voices are the phone's own (`Voices.english`, the list the panel's drawer shows). The
 * schema is the phone's (`tools/sync_settings.py`): a field the phone does not honour is not
 * in it, so the page never shows one that does nothing (Tai, 2026-09-24: the Voice setting
 * listed one voice, and the page showed the desktop's devices, folders and GPU switches).
 */
class SettingsApi(private val store: SettingsStore) {

    fun describe(): JSONObject = JSONObject().apply {
        put("settings", store.values())
        put("schema", JSONObject(store.schema))
        put("path", store.values().optString("data_dir"))
        put(
            "secrets",
            JSONObject().apply {
                for (name in store.secretNames()) put(name, store.secretSet(name))
            },
        )
        // The keys are in the app's own storage, which nothing else on the phone can read.
        put("keyring", true)
        put("key_envs", store.keyEnvs)
        put("devices", JSONObject().apply { put("inputs", JSONArray()); put("outputs", JSONArray()) })
        put("voices", JSONArray(Voices.english(PanelVoice.tts?.voicesPath)))
        put(
            "personas",
            JSONObject().apply {
                for ((name, pair) in com.sidecoach.coach.PERSONAS) put(name, pair.first)
            },
        )
    }

    fun update(body: JSONObject): JSONObject {
        val wanted = body.optJSONObject("settings") ?: return JSONObject().apply {
            put("ok", false)
            put("errors", JSONArray().put(JSONObject().put("msg", "no settings in the request")))
        }
        val before = store.values()
        // The phone owns these; the page shows them and cannot move them.
        wanted.put("data_dir", before.optString("data_dir"))
        wanted.put("tools_dir", before.optString("tools_dir"))
        store.save(wanted)
        val after = store.values()
        val changed = changed(before, after)
        val applies = applies(JSONObject(store.schema))
        val now = changed.filter { applies[it] == "now" }
        val later = changed.filter { applies[it] == "next" }
        val restart = changed.filter { applies[it] == "restart" }

        val stored = JSONObject()
        val secrets = body.optJSONObject("secrets")
        if (secrets != null) {
            val names = store.secretNames()
            for (name in secrets.keys()) {
                if (name !in names) continue
                val value = if (secrets.isNull(name)) null else secrets.optString(name)
                store.setSecret(name, value)
                stored.put(name, !value.isNullOrEmpty())
            }
        }

        // What the kept coach and engine can change, they do now; a "now" field with nothing
        // built to change is simply what the next build reads (`settings_api.update`).
        PanelWarm.applySettings(store, changed)

        PanelEngine.emit(
            "sys",
            "info",
            "settings saved" +
                (if (now.isNotEmpty()) "; applied now: " + words(now) else "") +
                (if (later.isNotEmpty()) "; at the next Start: " + words(later) else "") +
                (if (restart.isNotEmpty()) "; after a restart: " + words(restart) else "") +
                (if (changed.isEmpty()) "; nothing changed" else ""),
        )
        return JSONObject().apply {
            put("ok", true)
            put("path", after.optString("data_dir"))
            put("changed", JSONArray(changed))
            put("now", JSONArray(now))
            put("next", JSONArray(later))
            put("restart", JSONArray(restart))
            put("secrets", stored)
        }
    }

    private fun words(names: List<String>): String =
        names.joinToString(", ") { it.substringAfterLast('.').replace("_", " ") }

    /** "section.field" -> now | next | restart, from the schema's own marks. */
    private fun applies(schema: JSONObject): Map<String, String> {
        val out = HashMap<String, String>()
        val defs = schema.optJSONObject("\$defs") ?: JSONObject()
        val properties = schema.optJSONObject("properties") ?: JSONObject()
        for (name in properties.keys()) {
            val prop = properties.optJSONObject(name) ?: continue
            val ref = prop.optString("\$ref")
            if (ref.isNotEmpty()) {
                val model = defs.optJSONObject(ref.substringAfterLast('/')) ?: continue
                val fields = model.optJSONObject("properties") ?: continue
                for (field in fields.keys()) {
                    val one = fields.optJSONObject(field) ?: continue
                    out["$name.$field"] = one.optString("applies").ifEmpty { "next" }
                }
            } else {
                out[name] = prop.optString("applies").ifEmpty { "next" }
            }
        }
        return out
    }

    private fun changed(old: JSONObject, new: JSONObject, prefix: String = ""): List<String> {
        val out = ArrayList<String>()
        val names = linkedSetOf<String>()
        for (key in old.keys()) names.add(key)
        for (key in new.keys()) names.add(key)
        for (key in names) {
            val a = old.opt(key)
            val b = new.opt(key)
            if (a is JSONObject && b is JSONObject) {
                out.addAll(changed(a, b, "$prefix$key."))
            } else if (a?.toString() != b?.toString()) {
                out.add("$prefix$key")
            }
        }
        return out.sorted()
    }
}
