package com.sidecoach

import androidx.test.platform.app.InstrumentationRegistry
import org.junit.runner.Description
import org.junit.runner.notification.RunListener
import java.io.File

/**
 * The model files the tests load -- the engine's network, the voice's, the recogniser's --
 * moved into the app's own files at the start of a run (`tools/test.ps1` passes this listener).
 * The build the tests run is not debuggable (the `phone` build type), so `run-as` cannot put
 * them there: test.ps1 pushes what is missing into the app's external files (`push/`, which the
 * shell may write), and this, in the app's own process, moves each in. It then lists what the
 * app's files hold (`files.txt` beside `push/`), for the next run of test.ps1 to see what is
 * missing without asking the app.
 */
class FilesIn : RunListener() {

    override fun testRunStarted(description: Description?) {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val outside = context.getExternalFilesDir(null) ?: return
        val inside = context.filesDir
        val pushed = File(outside, "push")
        for (file in pushed.walkTopDown().filter { it.isFile }.toList()) {
            val target = File(inside, file.relativeTo(pushed).path)
            target.parentFile?.mkdirs()
            // Whole or not at all: a copy cut short is never taken for the file.
            val partial = File(target.path + ".part")
            file.inputStream().use { input -> partial.outputStream().use { input.copyTo(it) } }
            if (partial.length() == file.length() && partial.renameTo(target)) file.delete() else partial.delete()
        }
        // The files test.ps1 provides: the network at the top, the models under models/.
        val held = inside.listFiles { f -> f.isFile }.orEmpty().toList() +
            File(inside, "models").walkTopDown().filter { it.isFile }.toList()
        File(outside, "files.txt").writeText(
            held.joinToString("") { "${it.relativeTo(inside).path.replace('\\', '/')} ${it.length()}\n" },
        )
    }
}
