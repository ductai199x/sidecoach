package com.sidecoach.setup

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File
import java.io.IOException
import java.net.InetAddress
import java.net.ServerSocket
import java.nio.file.Files
import java.security.MessageDigest

/**
 * The voice's files are checked against their sha256 as they come, not by their size alone: a
 * body of the right length that is not the file was kept as the voice (audit 2026-10-07, S4).
 * The desktop's `test_a_voice_file_that_came_wrong_is_not_taken` and
 * `test_every_voice_file_has_its_checksum`, on a local server.
 */
class KokoroChecksumTest {

    @Test
    fun everyVoiceFileHasItsChecksum() {
        assertEquals(KOKORO_FILES.keys, KOKORO_SHA256.keys)
        assertTrue(KOKORO_SHA256.values.all { it.length == 64 })
    }

    @Test
    fun aVoiceFileThatCameWrongIsNotTaken() {
        val full = ByteArray(1024) { (it % 256).toByte() }
        val wrong = full.reversedArray() // the right size, not the file
        val server = ServerSocket(0, 8, InetAddress.getByName("127.0.0.1"))
        Thread {
            while (true) {
                val client = try { server.accept() } catch (_: Exception) { return@Thread }
                client.use {
                    val input = it.getInputStream()
                    val head = StringBuilder()
                    while (!head.endsWith("\r\n\r\n")) {
                        val ch = input.read()
                        if (ch < 0) break
                        head.append(ch.toChar())
                    }
                    val out = it.getOutputStream()
                    out.write("HTTP/1.1 200 OK\r\nContent-Length: ${wrong.size}\r\nConnection: close\r\n\r\n".toByteArray())
                    out.write(wrong)
                    out.flush()
                }
            }
        }.apply { isDaemon = true }.start()
        val dir = Files.createTempDirectory("kokoro-sha").toFile()
        try {
            val target = File(dir, "voices-v1.0.bin")
            val sha = MessageDigest.getInstance("SHA-256").digest(full).joinToString("") { "%02x".format(it) }
            try {
                download("http://127.0.0.1:${server.localPort}/v", target, expectSha256 = sha, expectSize = full.size.toLong())
                fail("a body that is not the file was taken")
            } catch (exc: IOException) {
                assertTrue(exc.message, "checksum" in exc.message.orEmpty())
            }
            assertFalse(target.exists())
            assertFalse(File(dir, "voices-v1.0.bin.part").exists())
        } finally {
            server.close()
            dir.deleteRecursively()
        }
    }
}
