package com.sidecoach

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sidecoach.analysis.EngineAssets
import com.sidecoach.bench.EmbedderBench
import com.sidecoach.capture.CaptureService
import com.sidecoach.capture.CaptureStats
import com.sidecoach.vision.GameWatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Spike A's screen: start the capture, watch what it costs, stop it.
 *
 * The real app's home screen comes with the pipeline; this one exists to answer whether
 * the phone can hand us frames for half an hour without heating up or flattening the
 * battery. The numbers it shows are the ones written to the log file.
 */
class MainActivity : ComponentActivity() {

    private lateinit var projections: MediaProjectionManager

    private val consent = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val data = result.data
        if (result.resultCode == Activity.RESULT_OK && data != null) {
            // The token is good for this one session; the service takes it from here.
            CaptureService.start(this, result.resultCode, data)
        }
    }

    private val notifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        // Denied, the capture still runs: Android shows its own chip either way. The
        // service's own notification is the one the person taps to stop it.
        askForScreen()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        projections = getSystemService(MediaProjectionManager::class.java)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    val state by CaptureService.capture.collectAsStateWithLifecycle()
                    Home(
                        running = state.running,
                        preview = state.preview,
                        reading = state.reading,
                        latest = state.latest,
                        logPath = state.logPath,
                        error = state.error,
                        onStart = ::startCapture,
                        onStop = { CaptureService.stop(this) },
                    )
                }
            }
        }
    }

    private fun startCapture() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) askForScreen() else notifications.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    /** The system's own consent dialog, which must be answered for every session. */
    private fun askForScreen() {
        consent.launch(projections.createScreenCaptureIntent())
    }
}

/** The game as the watcher has it: the board, the moves so far, and what it costs. */
@Composable
private fun Board(watch: GameWatcher.Watch) {
    if (!watch.boardFound) {
        Text("Looking for a board\u2026", style = MaterialTheme.typography.bodyMedium)
        return
    }
    val fen = watch.fen
    if (fen == null) {
        Text(
            "Board found at ${watch.region?.x},${watch.region?.y}, cell " +
                "${"%.0f".format(watch.cellPx)} px. Waiting for the start position to learn " +
                "the pieces.",
            style = MaterialTheme.typography.bodyMedium,
        )
        return
    }
    val placement = fen.substringBefore(' ')
    val turn = if (fen.contains(" w ")) "white" else "black"
    Text(
        when (watch.state) {
            com.sidecoach.events.ScreenState.GAME_OVER -> "Game over ${watch.finished ?: ""}"
            com.sidecoach.events.ScreenState.COVERED -> "The board is covered"
            else -> "Ply ${watch.ply} \u00b7 $turn to move"
        },
        style = MaterialTheme.typography.titleMedium,
    )
    Text(
        placement.split('/').joinToString("\n") { rank ->
            rank.flatMap { ch -> if (ch.isDigit()) List(ch - '0') { '.' } else listOf(ch) }
                .joinToString(" ")
        },
        fontFamily = FontFamily.Monospace,
        style = MaterialTheme.typography.bodyLarge,
    )
    if (watch.pgn.isNotEmpty()) {
        Text(watch.pgn, style = MaterialTheme.typography.bodyMedium)
    }
    if (watch.verdicts.isNotEmpty()) {
        Text(watch.verdicts.takeLast(6).joinToString("  \u00b7  "), style = MaterialTheme.typography.bodyMedium)
    }
    Text(
        "${watch.unsure} unsure \u00b7 ${watch.fresh} cells embedded \u00b7 " +
            "${if (watch.verified) "verified" else "held"} \u00b7 " +
            "${if (watch.changed) "changed" else "still"} \u00b7 read in ${watch.readMs} ms" +
            (if (watch.engineOn) "" else " \u00b7 no engine"),
        style = MaterialTheme.typography.bodySmall,
    )
}

/**
 * The engine's network: 98 MB, fetched once. The desktop downloads the engine on first run
 * and this is the same bargain the other way round -- the engine ships, the network does not.
 */
@Composable
private fun EngineNet() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val state by EngineAssets.net.collectAsStateWithLifecycle()
    LaunchedEffect(Unit) { EngineAssets.found(context) }
    when (val now = state) {
        is EngineAssets.State.Ready ->
            Text(
                "The engine is ready (${now.file.length() / 1_000_000} MB network).",
                style = MaterialTheme.typography.bodySmall,
            )
        is EngineAssets.State.Fetching -> {
            val done = if (now.total > 0) now.bytes * 100 / now.total else 0
            Text(
                "Fetching the engine's network: $done% of ${now.total / 1_000_000} MB",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
        else -> {
            if (now is EngineAssets.State.Failed) {
                Text("The network did not download: ${now.why}", style = MaterialTheme.typography.bodySmall)
            }
            Text(
                "Without the engine's network the game is watched and followed, but nothing " +
                    "is assessed. It is 98 MB and is fetched once.",
                style = MaterialTheme.typography.bodySmall,
            )
            Button(onClick = {
                scope.launch { withContext(Dispatchers.IO) { EngineAssets.fetch(context) } }
            }) {
                Text("Get the engine's network")
            }
        }
    }
}

@Composable
private fun Home(
    running: Boolean,
    preview: Bitmap?,
    reading: GameWatcher.Watch?,
    latest: CaptureStats.Snapshot?,
    logPath: String?,
    error: String?,
    onStart: () -> Unit,
    onStop: () -> Unit,
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Sidecoach", style = MaterialTheme.typography.headlineMedium)
        Text(
            if (running) "Watching the screen. Switch to your chess app; the position it reads is here."
            else "Watches the board on your screen and reads the position.",
            style = MaterialTheme.typography.bodyMedium,
        )
        Button(onClick = if (running) onStop else onStart) {
            Text(if (running) "Stop" else "Start watching")
        }
        error?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
        EngineNet()
        reading?.let { Board(it) }
        preview?.let {
            Text("What it sees", style = MaterialTheme.typography.titleMedium)
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "the screen as the coach reads it",
                contentScale = ContentScale.FillWidth,
                modifier = Modifier.fillMaxWidth(),
            )
        }
        if (latest != null) {
            Text("Last ten seconds", style = MaterialTheme.typography.titleMedium)
            Text(
                """
                frames ${latest.frames} (${"%.1f".format(latest.fps)} fps), ${latest.dropped} skipped
                cpu ${"%.1f".format(latest.cpuPercent)} % of a core
                battery ${latest.batteryPercent} %, ${latest.batteryMa} mA · thermal ${latest.thermal}
                """.trimIndent(),
                style = MaterialTheme.typography.bodySmall,
            )
        }
        logPath?.let { Text("Log: $it", style = MaterialTheme.typography.bodySmall) }

        // Spike B's first question: what a board costs the piece net on this phone.
        val context = LocalContext.current
        val scope = rememberCoroutineScope()
        var bench by remember { mutableStateOf<String?>(null) }
        var benching by remember { mutableStateOf(false) }
        Button(
            enabled = !benching,
            onClick = {
                benching = true
                bench = "embedding 64 cells, ten times over…"
                scope.launch {
                    val results = withContext(Dispatchers.Default) { EmbedderBench.run(context) }
                    bench = results.joinToString(separator = "\n") { it.toString() }
                    benching = false
                }
            },
        ) {
            Text("Time the piece net")
        }
        bench?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
    }
}
