package com.sidecoach.debug

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.widget.ImageView
import androidx.activity.ComponentActivity

/**
 * A rendered board, full screen, for testing the vision without a chess app.
 *
 * One board from the app's own assets (tools/make_fixtures.py renders them with the desktop
 * renderer, so what the capture reads here is what the Python suite reads), or a game: the
 * frames of tools/make_game.py, shown one after another so the whole pipeline runs against a
 * game on the device. Both live in the debug source set and never reach a release build.
 *
 * Debug only.
 *
 *     adb shell am start -n com.sidecoach/.debug.BoardActivity -e board start-green
 *     adb shell am start -n com.sidecoach/.debug.BoardActivity  *         -e game /sdcard/Android/data/com.sidecoach/files/game --ei hold 2500
 */
class BoardActivity : ComponentActivity() {

    private lateinit var view: ImageView
    private val handler = Handler(Looper.getMainLooper())
    private var frames: List<String> = emptyList()
    private var at = 0
    private var holdMs = 2500L

    /**
     * The start position is held longer than the moves, as it is in a real game while
     * both players get ready: the tracker needs a frame or two that is not a screen
     * sliding into view before it has a lock to read the position with.
     */
    private var leadMs = 8000L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        view = ImageView(this)
        view.scaleType = ImageView.ScaleType.FIT_CENTER
        view.setBackgroundColor(0xFF101010.toInt())
        // A game shown to the watcher runs for minutes: the screen must not sleep under it.
        view.keepScreenOn = true
        setContentView(view)
        show(intent)
    }

    // The activity is singleTop, so `am start` with another board arrives here rather than
    // being swallowed as a duplicate intent.
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        show(intent)
    }

    override fun onDestroy() {
        handler.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    private fun show(intent: Intent) {
        handler.removeCallbacksAndMessages(null)
        if (intent.hasExtra("game")) {
            holdMs = intent.getIntExtra("hold", 2500).toLong()
            leadMs = intent.getIntExtra("lead", 8000).toLong()
            val folder = intent.getStringExtra("game").orEmpty().ifEmpty { "game" }
            frames = (assets.list(folder) ?: emptyArray()).filter { it.endsWith(".png") }
                .sorted().map { "$folder/$it" }
            at = 0
            Log.i(TAG, "playing ${frames.size} frames of $folder every $holdMs ms")
            step()
            return
        }
        val name = intent.getStringExtra("board") ?: "start-green"
        runCatching {
            assets.open("debug/$name.png").use { view.setImageBitmap(BitmapFactory.decodeStream(it)) }
        }
    }

    private fun step() {
        if (at >= frames.size) {
            Log.i(TAG, "the game is over; the last frame stays up")
            return
        }
        val name = frames[at]
        runCatching {
            assets.open(name).use { view.setImageBitmap(BitmapFactory.decodeStream(it)) }
        }
        Log.i(TAG, "frame $name")
        val wait = if (at == 0) leadMs else holdMs
        at += 1
        handler.postDelayed({ step() }, wait)
    }

    private companion object {
        const val TAG = "sidecoach.debug"
    }
}
