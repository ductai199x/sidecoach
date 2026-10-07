package com.sidecoach

import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream

/**
 * The test fixtures (androidTest/assets) and the app's own assets (main/assets), as the host's
 * unit tests read them: from the source tree, whose paths Gradle passes (app/build.gradle.kts).
 * The device's tests have the same object over the APKs' AssetManagers (androidTest's
 * TestAssets.kt): a test reads `TestAssets.text("chess/positions.json")` on either side.
 */
object TestAssets {
    private val fixtures = root("sidecoach.assets")
    private val app = root("sidecoach.appAssets")

    /** A fixture of the test APK (`androidTest/assets`). */
    fun open(path: String): InputStream = read(fixtures, path)

    fun text(path: String): String = open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }

    /** The names in a fixture folder, as AssetManager.list gives them (none for a missing folder). */
    fun list(path: String): List<String> = File(fixtures, path).list().orEmpty().sorted()

    /** An asset of the app itself (`main/assets`), as the app's AssetManager gives it. */
    fun openApp(path: String): InputStream = read(app, path)

    private fun read(root: File, path: String): InputStream {
        val file = File(root, path)
        // As AssetManager.open: a missing asset is an error, never an empty stream.
        if (!file.isFile) throw FileNotFoundException(path)
        return file.inputStream().buffered()
    }

    private fun root(property: String): File {
        val path = System.getProperty(property)
            ?: error("$property is not set: run the tests through Gradle (:app:testPhoneUnitTest)")
        return File(path).also { check(it.isDirectory) { "$property: no such directory $it" } }
    }
}

/** A test's own report line (counts, times), to standard output on the host: logcat's on the device. */
object TestLog {
    fun i(tag: String, message: String) {
        println("I/$tag: $message")
    }
}
