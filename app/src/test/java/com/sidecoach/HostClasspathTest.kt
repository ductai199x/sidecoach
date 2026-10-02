package com.sidecoach

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What the host's unit tests run on. org.json is Android's (AOSP libcore's, the :libcore-json
 * module), not android.jar's stubs, which throw, nor json.org's or the 2013 Android snapshot on
 * Maven, which order keys by hash: the parity tests compare JSON the code builds as the phone
 * builds it. And the rest of android.jar fails loudly, so a test that needs the phone cannot pass
 * here on a stub's default answer.
 */
class HostClasspathTest {

    @Test
    fun orgJsonIsAndroidsOwn() {
        val source = JSONObject::class.java.protectionDomain.codeSource.location.toString()
        assertTrue(source, "libcore-json" in source)
        // Android's JSONObject keeps the order of insertion; json.org's is a HashMap.
        val keys = listOf("zeta", "alpha", "mu", "beta", "omega", "delta", "kappa", "gamma")
        val ordered = JSONObject()
        keys.forEachIndexed { i, key -> ordered.put(key, i) }
        assertEquals(keys, ordered.keys().asSequence().toList())
        assertEquals("""{"zeta":0,"alpha":1,"mu":2,"beta":3,"omega":4,"delta":5,"kappa":6,"gamma":7}""", ordered.toString())
        // As the phone writes them: a whole double as an integer, a slash escaped.
        assertEquals("""{"a":1}""", JSONObject().put("a", 1.0).toString())
        assertEquals("""{"a":0.1,"b":-2.5E-7}""", JSONObject().put("a", 0.1).put("b", -2.5e-7).toString())
        assertEquals("""{"url":"a\/b"}""", JSONObject().put("url", "a/b").toString())
        assertEquals("""[1,"x",null,true]""", JSONArray().put(1).put("x").put(JSONObject.NULL).put(true).toString())
        // Read back as Android reads: an int an Integer, a fraction a Double.
        val read = JSONObject("""{"i": 3, "l": 3000000000, "d": 1.5}""")
        assertEquals(Integer::class.java, read.get("i")::class.javaObjectType)
        assertEquals(java.lang.Long::class.java, read.get("l")::class.javaObjectType)
        assertEquals(java.lang.Double::class.java, read.get("d")::class.javaObjectType)
    }

    @Test
    fun theAndroidRuntimeIsNotHereAndSaysSo() {
        // No isReturnDefaultValues: a stub throws rather than answer 0, null or false.
        val log = assertThrows(RuntimeException::class.java) { android.util.Log.i("t", "m") }
        assertTrue(log.message, "not mocked" in log.message.orEmpty())
        val clock = assertThrows(RuntimeException::class.java) { android.os.SystemClock.elapsedRealtime() }
        assertTrue(clock.message, "not mocked" in clock.message.orEmpty())
    }
}
