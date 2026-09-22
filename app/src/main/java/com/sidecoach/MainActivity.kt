package com.sidecoach

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.sidecoach.capture.CaptureService
import com.sidecoach.ui.LocalServer
import com.sidecoach.ui.PanelApi
import com.sidecoach.ui.PanelEngine
import com.sidecoach.ui.SettingsApi
import com.sidecoach.ui.SettingsStore
import com.sidecoach.ui.SetupApi
import java.io.File

/**
 * The panel, in a window.
 *
 * The page is the desktop's own -- the SvelteKit build in `sidecoach/ui/web`, shipped here as
 * the asset folder `web/` and kept in step by `tools/sync_web.py` -- served from a local HTTP
 * server and shown in a WebView. It is the same frontend on both platforms, talking to the
 * same API over the same transport, and its stylesheet already has the stacked layout it
 * describes as being for "a phone, a tablet held upright, a narrow window".
 *
 * The one thing the phone does differently is Start: a computer begins reading the screen on
 * its own word, and Android wants the person's consent for the projection first. So the
 * page's Start reaches back here, the system asks, and the service takes the token from
 * there; the page sees `running` turn true on its next poll, as it would on a computer.
 */
class MainActivity : ComponentActivity() {

    private lateinit var projections: MediaProjectionManager
    private var server: LocalServer? = null
    private var web: WebView? = null

    private val consent = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val data = result.data
        if (result.resultCode == Activity.RESULT_OK && data != null) {
            // The token is good for this one session; the service takes it from here.
            CaptureService.start(this, result.resultCode, data)
        } else {
            PanelEngine.emit("sys", "warn", "the screen was not shared, so nothing is watched")
        }
    }

    private val notifications = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) {
        // Denied, the capture still runs: Android shows its own chip either way. The
        // service's own notification is the one the person taps to stop it.
        askForScreen()
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        projections = getSystemService(MediaProjectionManager::class.java)

        PanelEngine.onStart = { runOnUiThread { startCapture() } }
        PanelEngine.onStop = { runOnUiThread { CaptureService.stop(this) } }

        val store = SettingsStore(this)
        val port = LocalServer(
            assets,
            api = PanelApi(
                versionName(),
                SetupApi(this),
                SettingsApi(store),
                themes = File(filesDir, "themes"),
            ),
        ).also { server = it }.start()

        val view = WebView(this)
        web = view
        view.settings.apply {
            javaScriptEnabled = true
            // The page remembers its pane sizes and switches in localStorage.
            domStorageEnabled = true
            // Honour the page's own <meta name="viewport">: without this the WebView lays
            // the page out at whatever width its widest element wants, and the right edge
            // of the settings page was off the screen with no way to scroll to it.
            useWideViewPort = true
            loadWithOverviewMode = true
            // Everything it loads comes from the local server; nothing else is reachable.
            allowFileAccess = false
            allowContentAccess = false
            mediaPlaybackRequiresUserGesture = false
        }
        view.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView?,
                request: WebResourceRequest?,
            ): Boolean {
                val url = request?.url ?: return false
                // The panel's own routes stay in the window; a link out goes to the browser.
                if (url.host == "127.0.0.1" || url.host == "localhost") return false
                runCatching {
                    startActivity(Intent(Intent.ACTION_VIEW, url))
                }.onFailure { Log.i(TAG, "nothing opens $url") }
                return true
            }

            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?,
            ) {
                Log.w(TAG, "the page could not load ${request?.url}: ${error?.description}")
            }
        }
        view.webChromeClient = object : WebChromeClient() {
            override fun onConsoleMessage(message: ConsoleMessage): Boolean {
                Log.i(TAG, "page: ${message.message()} (${message.sourceId()}:${message.lineNumber()})")
                return true
            }
        }
        if (0 != applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) {
            // chrome://inspect reaches it, as it reaches the desktop's window.
            WebView.setWebContentsDebuggingEnabled(true)
        }
        setContentView(view)
        view.loadUrl("http://127.0.0.1:$port/")

        // Back walks the page's own history -- out of the settings, back to the panel --
        // and leaves the app only when there is nowhere left to go.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                val page = web
                if (page != null && page.canGoBack()) {
                    page.goBack()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })
    }

    override fun onDestroy() {
        PanelEngine.onStart = null
        PanelEngine.onStop = null
        web?.destroy()
        web = null
        server?.stop()
        server = null
        super.onDestroy()
    }

    private fun startCapture() {
        val granted = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.POST_NOTIFICATIONS,
        ) == PackageManager.PERMISSION_GRANTED
        if (granted) askForScreen() else notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    /** The system's own consent dialog, which must be answered for every session. */
    private fun askForScreen() {
        consent.launch(projections.createScreenCaptureIntent())
    }

    private fun versionName(): String = try {
        packageManager.getPackageInfo(packageName, 0).versionName ?: "0"
    } catch (_: Exception) {
        "0"
    }

    companion object {
        private const val TAG = "sidecoach.panel"
    }
}
