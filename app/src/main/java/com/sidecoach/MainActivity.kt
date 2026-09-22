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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sidecoach.capture.CaptureService
import com.sidecoach.capture.CaptureStats

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

@Composable
private fun Home(
    running: Boolean,
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
            if (running) "Watching the screen. Switch to your chess app; come back for the numbers."
            else "Spike A: the screen read at ${CaptureService.TARGET_FPS} frames a second, half size.",
            style = MaterialTheme.typography.bodyMedium,
        )
        Button(onClick = if (running) onStop else onStart) {
            Text(if (running) "Stop" else "Start watching")
        }
        error?.let { Text(it, style = MaterialTheme.typography.bodyMedium) }
        if (latest != null) {
            Text("Last ten seconds", style = MaterialTheme.typography.titleMedium)
            Text(
                """
                running ${"%.0f".format(latest.elapsedS)} s
                frames ${latest.frames} (${"%.1f".format(latest.fps)} fps), ${latest.dropped} skipped
                interval p50 ${latest.intervalP50} ms, p95 ${latest.intervalP95}, max ${latest.intervalMax}
                copy p50 ${latest.copyP50} ms, p95 ${latest.copyP95}, frame ${latest.frameKb} KB
                cpu ${"%.1f".format(latest.cpuPercent)} % of a core
                battery ${latest.batteryPercent} %, ${latest.batteryMa} mA
                thermal ${latest.thermal}
                """.trimIndent(),
                style = MaterialTheme.typography.bodySmall,
            )
        }
        logPath?.let { Text("Log: $it", style = MaterialTheme.typography.bodySmall) }
    }
}
