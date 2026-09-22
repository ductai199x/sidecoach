package com.sidecoach.debug

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.ImageView
import androidx.activity.ComponentActivity

/**
 * A rendered board, full screen, for testing the vision without a chess app.
 *
 * The boards are the desktop renderer's own (tools/make_fixtures.py), so what the capture
 * reads here is what the Python suite reads. Debug only: it is how the emulator gets a
 * board on its screen, and how a phone can be pointed at a known position.
 *
 *     adb shell am start -n com.sidecoach/.debug.BoardActivity -e board start-green
 */
class BoardActivity : ComponentActivity() {

    private lateinit var view: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        view = ImageView(this)
        view.scaleType = ImageView.ScaleType.FIT_CENTER
        view.setBackgroundColor(0xFF101010.toInt())
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

    private fun show(intent: Intent) {
        val name = intent.getStringExtra("board") ?: "start-green"
        runCatching {
            assets.open("debug/$name.png").use { view.setImageBitmap(BitmapFactory.decodeStream(it)) }
        }
    }
}
