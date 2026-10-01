package com.sidecoach

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.widget.FrameLayout
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.JsResult
import android.app.AlertDialog
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.sidecoach.capture.CaptureService
import com.sidecoach.capture.RecordingNotice
import com.sidecoach.ui.Captions
import com.sidecoach.ui.GamesApi
import com.sidecoach.ui.LocalServer
import com.sidecoach.ui.PanelApi
import com.sidecoach.ui.PanelEngine
import com.sidecoach.ui.PanelListen
import com.sidecoach.ui.PanelWarm
import com.sidecoach.ui.SettingsApi
import com.sidecoach.ui.PuzzleRooms
import com.sidecoach.ui.ReviewRooms
import com.sidecoach.ui.SettingsStore
import com.sidecoach.ui.SetupApi
import com.sidecoach.ui.StatsApi
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

    /** The microphone, asked for when a listening mode first needs it (see `PanelListen`). */
    private val microphone = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> PanelListen.permitted(granted) }

    /**
     * The page's file inputs -- Settings > Board's own pieces, Your games' .pgn file -- open
     * Android's picker, and what was chosen goes back to the page. A WebView opens nothing for
     * a file input on its own: the button did nothing at all on the phone (Tai, 2026-10-01).
     */
    private var filesWanted: android.webkit.ValueCallback<Array<android.net.Uri>>? = null
    private val pickFiles = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val data = result.data
        val clip = data?.clipData
        val chosen = when {
            result.resultCode != Activity.RESULT_OK || data == null -> null
            clip != null -> Array(clip.itemCount) { clip.getItemAt(it).uri }
            data.data != null -> arrayOf(data.data!!)
            else -> null
        }
        filesWanted?.onReceiveValue(chosen)
        filesWanted = null
    }

    /** Android's document picker for a file input: its types, one file or several. */
    private fun pickerFor(params: WebChromeClient.FileChooserParams): Intent {
        val types = params.acceptTypes.flatMap { it.split(",") }.map { it.trim() }.filter { "/" in it }.distinct()
        return Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = types.singleOrNull() ?: "*/*"
            if (types.size > 1) putExtra(Intent.EXTRA_MIME_TYPES, types.toTypedArray())
            if (params.mode == WebChromeClient.FileChooserParams.MODE_OPEN_MULTIPLE) putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        // Edge to edge on every version, as Android 15 and later always are: the bars are
        // transparent, with light icons for the dark pages, and the frame paints under
        // them in the page's own colours (see `framed`). Without this, Android 14 and
        // earlier painted them the theme's black.
        enableEdgeToEdge(
            statusBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
            navigationBarStyle = SystemBarStyle.dark(Color.TRANSPARENT),
        )
        super.onCreate(savedInstanceState)
        projections = getSystemService(MediaProjectionManager::class.java)

        PanelEngine.onStart = { runOnUiThread { startCapture() } }
        PanelEngine.onStop = { runOnUiThread { CaptureService.stop(this) } }
        PanelListen.askPermission = { runOnUiThread { microphone.launch(Manifest.permission.RECORD_AUDIO) } }

        val store = SettingsStore(this)
        val ports = getSharedPreferences("panel", MODE_PRIVATE)
        val port = LocalServer(
            assets,
            api = PanelApi(
                versionName(),
                // One setup for the app, as the desktop's server has one: a turn of the phone
                // rebuilds this activity, and a download under way must keep its status.
                SetupApi.shared(this),
                SettingsApi(store),
                files = filesDir,
                rooms = { PuzzleRooms.shared(applicationContext) },
                // The app's own folder on shared storage, where `adb pull` reaches a dump.
                dumps = File(getExternalFilesDir(null) ?: filesDir, "dumps"),
                games = GamesApi(filesDir, store),
                stats = StatsApi(filesDir),
                // One set of review rooms for the app, as the puzzle rooms: a review under way
                // outlives the window being rebuilt (a turn of the phone).
                reviews = { ReviewRooms.shared(applicationContext) },
            ),
        ).also { server = it }.start(preferred = ports.getInt(PORT_KEY, 0))
        // The same port next time, so the page's origin -- and its saved sizes and switches
        // with it -- stays the same from one launch to the next.
        ports.edit().putInt(PORT_KEY, port).apply()

        val view = WebView(this)
        web = view
        // The ground the page paints on, not the WebView's white, for the moment before it
        // first paints: the splash card's on a launch, the header's when already warm.
        view.setBackgroundColor(getColor(if (PanelWarm.done) R.color.panel else R.color.splash_ground))
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

            // The page's own questions ("Reset the profile? ...") as the app's dialogs: the
            // WebView's default one is headed with the local server's address.
            override fun onJsAlert(view: WebView?, url: String?, message: String?, result: JsResult): Boolean {
                AlertDialog.Builder(this@MainActivity)
                    .setMessage(message)
                    .setPositiveButton(android.R.string.ok) { _, _ -> result.confirm() }
                    .setOnCancelListener { result.confirm() }
                    .show()
                return true
            }

            override fun onShowFileChooser(
                view: WebView?,
                callback: android.webkit.ValueCallback<Array<android.net.Uri>>,
                params: FileChooserParams,
            ): Boolean {
                // One picker at a time: one still open gets nothing back.
                filesWanted?.onReceiveValue(null)
                filesWanted = callback
                return try {
                    pickFiles.launch(pickerFor(params))
                    true
                } catch (e: android.content.ActivityNotFoundException) {
                    Log.w(TAG, "no file picker on this phone: $e")
                    filesWanted = null
                    false
                }
            }

            override fun onJsConfirm(view: WebView?, url: String?, message: String?, result: JsResult): Boolean {
                AlertDialog.Builder(this@MainActivity)
                    .setMessage(message)
                    .setPositiveButton(android.R.string.ok) { _, _ -> result.confirm() }
                    .setNegativeButton(android.R.string.cancel) { _, _ -> result.cancel() }
                    .setOnCancelListener { result.cancel() }
                    .show()
                return true
            }
        }
        if (BuildConfig.DEV_TOOLS) {
            // chrome://inspect reaches it, as it reaches the desktop's window: in a debug build,
            // and in the phone's build, which was a debug build until 2026-10-01.
            WebView.setWebContentsDebuggingEnabled(true)
        }
        setContentView(framed(view))
        // Everything a session needs loads now, so Start only starts the screen; the splash
        // shows it land. Once: an app already warm opens on the panel.
        PanelWarm.warm(this)
        view.loadUrl("http://127.0.0.1:$port/" + if (PanelWarm.done) "" else "splash")

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

    /**
     * The page on the whole screen, under the system bars, told their heights: the local
     * server writes them into each page as CSS variables before it paints, and a change
     * reaches the page at once. The side insets (a camera cutout in landscape) and the
     * keyboard are taken here instead: the keyboard resizes the page, so the ask box rises
     * above it, and the navigation bar is under it then.
     */
    private fun framed(page: WebView): FrameLayout {
        val frame = FrameLayout(this).apply { setBackgroundColor(getColor(R.color.splash_ground)) }
        frame.addView(page, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))
        ViewCompat.setOnApplyWindowInsetsListener(frame) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout(),
            )
            val keyboard = insets.getInsets(WindowInsetsCompat.Type.ime()).bottom
            (page.layoutParams as FrameLayout.LayoutParams).setMargins(bars.left, 0, bars.right, keyboard)
            page.requestLayout()
            val density = resources.displayMetrics.density
            val top = Math.round(bars.top / density)
            val bottom = if (keyboard > 0) 0 else Math.round(bars.bottom / density)
            val style = "--inset-top: ${top}px; --inset-bottom: ${bottom}px"
            if (style != server?.htmlStyle) {
                server?.htmlStyle = style
                page.evaluateJavascript(
                    "(() => { const s = document.documentElement.style; " +
                        "s.setProperty('--inset-top', '${top}px'); " +
                        "s.setProperty('--inset-bottom', '${bottom}px'); })()",
                    null,
                )
            }
            WindowInsetsCompat.CONSUMED
        }
        return frame
    }

    // The coach may hear while its panel is in front; behind another app, only a session holds
    // the microphone (`PanelListen`).
    override fun onStart() {
        super.onStart()
        PanelListen.visible(true)
        Captions.panelInFront(true)
    }

    override fun onStop() {
        PanelListen.visible(false)
        Captions.panelInFront(false)
        super.onStop()
    }

    override fun onDestroy() {
        // Quit (backed out of, or swiped from the recents): the caption box goes too.
        if (isFinishing) Captions.quit()
        PanelEngine.onStart = null
        PanelEngine.onStop = null
        PanelListen.askPermission = null
        web?.destroy()
        web = null
        server?.stop()
        server = null
        super.onDestroy()
    }

    private fun startCapture() {
        // A chess app that can be told it is being recorded is named first, and the screen
        // is shared only on Continue (`RecordingNotice`).
        val installed = RecordingNotice.installed(this)
        // Also in the panel's own log, which keeps it: the system's is a few hundred kilobytes
        // and drops the chattiest app's lines first (a Start's line was gone within minutes).
        val found = RecordingNotice.survey(installed)
        Log.i(TAG, "recording notice: $found")
        PanelEngine.emit("sys", "debug", "recording notice: $found")
        val watching = RecordingNotice.unanswered(this, installed)
        if (watching.isNotEmpty()) {
            RecordingNotice.ask(this, watching, proceed = { shareScreen() }, cancelled = {
                PanelEngine.emit(
                    "sys",
                    "info",
                    "not started, as you chose: ${RecordingNotice.names(watching)} can tell when it is recorded",
                )
            })
            return
        }
        shareScreen()
    }

    private fun shareScreen() {
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

        /** Where the panel's port is kept between launches. */
        private const val PORT_KEY = "port"
    }
}
