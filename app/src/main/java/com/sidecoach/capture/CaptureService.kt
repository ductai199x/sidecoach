package com.sidecoach.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import android.util.Log
import android.view.WindowManager
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.sidecoach.MainActivity
import com.sidecoach.SidecoachApp
import com.sidecoach.R
import com.sidecoach.ui.PanelEngine
import com.sidecoach.vision.GameWatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The screen, read in a foreground service.
 *
 * The order the platform insists on (Android 14 and up): the activity asks for consent,
 * hands the result here, the service goes to the foreground with the mediaProjection
 * type, and only then takes the projection, registers its callback -- without which
 * `createVirtualDisplay` throws -- and mirrors the display into an [ImageReader]. The
 * person can stop it from the status bar chip at any moment, and the system stops it on
 * the lock screen; [MediaProjection.Callback.onStop] is where that arrives.
 *
 * The frames are not kept. Spike A only measures them: the interval between them, the
 * cost of copying one out, this process's CPU, the battery and the thermal status (see
 * [CaptureStats]). The vision pipeline comes later, in its own module.
 */
class CaptureService : Service() {

    private var projection: MediaProjection? = null
    private var display: VirtualDisplay? = null
    private var reader: ImageReader? = null
    private var thread: HandlerThread? = null
    private var handler: Handler? = null
    private var stats: CaptureStats? = null
    private var lastFrameAt = 0L
    private var buffer: ByteArray? = null
    // The frame as pixels we own, reused: the reader's buffer is handed back the moment
    // the image is closed, so nothing may be kept from it.
    private var frame: Bitmap? = null
    private var preview: Bitmap? = null
    private var lastPreviewAt = 0L
    // The vision: the board found, its theme learned, the position read, the game
    // followed. One watcher for the session, on its own thread.
    private var vision: GameWatcher? = null
    private var visionThread: HandlerThread? = null
    private var visionHandler: Handler? = null
    private var reading = false
    private val frameLock = Any()
    private var lastReadAt = 0L
    private var lastPlacement: String? = null
    private var lastOutcomeAt = 0L

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, 0) ?: 0
        val data: Intent? = if (Build.VERSION.SDK_INT >= 33) {
            intent?.getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent?.getParcelableExtra(EXTRA_RESULT_DATA)
        }
        if (data == null) {
            Log.w(TAG, "started without a consent result; nothing to capture")
            stopSelf()
            return START_NOT_STICKY
        }

        // The foreground notification first: the projection may not be taken before the
        // service carries the mediaProjection type.
        ServiceCompat.startForeground(
            this,
            NOTIFICATION_ID,
            notification(),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
        )

        val manager = getSystemService(MediaProjectionManager::class.java)
        val projection = runCatching { manager.getMediaProjection(resultCode, data) }.getOrNull()
        if (projection == null) {
            Log.w(TAG, "the consent token was refused; it is good for one session only")
            state.value = State(error = "The screen-capture permission was not granted.")
            PanelEngine.emit("sys", "warn", "the screen-capture permission was not granted")
            stopSelf()
            return START_NOT_STICKY
        }
        this.projection = projection

        val worker = HandlerThread("capture").also { it.start() }
        thread = worker
        handler = Handler(worker.looper)

        projection.registerCallback(
            object : MediaProjection.Callback() {
                override fun onStop() {
                    // The chip, the lock screen, another app's projection, or us.
                    Log.i(TAG, "projection stopped")
                    stopSelf()
                }

                override fun onCapturedContentResize(width: Int, height: Int) {
                    resize(width, height)
                }
            },
            handler,
        )

        PanelEngine.running(true)
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        vision = GameWatcher(this, File(filesDir, "themes"))
        val visionWorker = HandlerThread("vision").also { it.start() }
        visionThread = visionWorker
        visionHandler = Handler(visionWorker.looper)
        stats = CaptureStats(this, File(filesDir, "spike/capture-$stamp.jsonl")).also {
            it.engineCpu = { vision?.engineCpuMs() ?: 0 }
        }
        start(projection)
        state.value = State(running = true, logPath = "spike/capture-$stamp.jsonl")
        return START_NOT_STICKY
    }

    private fun start(projection: MediaProjection) {
        val metrics = getSystemService(WindowManager::class.java).maximumWindowMetrics.bounds
        // Half the width and half the height: a quarter of the pixels, which is plenty
        // for a board that fills most of a phone screen, and a quarter of the copy.
        val width = (metrics.width() / SCALE) and 1.inv()
        val height = (metrics.height() / SCALE) and 1.inv()
        val density = resources.displayMetrics.densityDpi

        val reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, BUFFERS)
        this.reader = reader
        reader.setOnImageAvailableListener({ onFrame(it) }, handler)

        display = projection.createVirtualDisplay(
            "sidecoach",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            null,
            handler,
        )
        Log.i(TAG, "capturing ${width}x$height at ${TARGET_FPS} fps")
    }

    /** The screen turned, or the captured window changed size. */
    private fun resize(width: Int, height: Int) {
        val w = (width / SCALE) and 1.inv()
        val h = (height / SCALE) and 1.inv()
        val display = display ?: return
        runCatching {
            display.resize(w, h, resources.displayMetrics.densityDpi)
            reader?.surface?.let { /* the reader keeps its own size; a new one follows */ }
        }.onFailure { Log.w(TAG, "resize failed: $it") }
    }

    private fun onFrame(reader: ImageReader) {
        val image = runCatching { reader.acquireLatestImage() }.getOrNull() ?: return
        try {
            val now = SystemClock.elapsedRealtime()
            if (now - lastFrameAt < MIN_INTERVAL_MS) {
                stats?.skipped()
                return
            }
            lastFrameAt = now
            val began = SystemClock.elapsedRealtimeNanos()
            val plane = image.planes[0]
            val size = plane.rowStride * image.height
            val into = buffer?.takeIf { it.size == size } ?: ByteArray(size).also { buffer = it }
            // What the vision pipeline will pay on every frame: the pixels out of the
            // reader's buffer and into memory we own.
            plane.buffer.rewind()
            plane.buffer.get(into)
            val copyMs = ((SystemClock.elapsedRealtimeNanos() - began) / 1_000_000).toInt()
            stats?.frame(copyMs, size)
            keepTheFrame(image, plane.rowStride)
            keepForScreen(image.width, image.height)
        } catch (exc: Exception) {
            Log.w(TAG, "frame dropped: $exc")
        } finally {
            image.close()
        }
        // Only now, with the reader's buffer handed back: a board read with the image
        // still open starved the display's queue and the frames stopped coming.
        askForARead()
        val stats = stats ?: return
        if (stats.due()) {
            val snapshot = stats.take()
            state.value = state.value.copy(running = true, latest = snapshot)
        }
    }

    /**
     * The newest frame, kept for the vision. Every one of them: the display sends a frame
     * only when the screen changes, so the last one before a screen goes still is the one
     * the vision will study until it changes again. Sampling these at the preview's rate
     * left a board that appeared and stopped moving being read from the frame halfway
     * through the animation that brought it in, for as long as nothing else moved -- on a
     * quiet board, twenty-five seconds before it was found.
     *
     * The bitmap the reader's plane is copied into carries the row stride, so it is wider
     * than the screen; the extra column is beyond the board and costs nothing.
     */
    private fun keepTheFrame(image: android.media.Image, rowStride: Int) {
        val padded = rowStride / 4
        val height = image.height
        val full = frame?.takeIf { it.width == padded && it.height == height }
            ?: Bitmap.createBitmap(padded, height, Bitmap.Config.ARGB_8888).also { frame = it }
        image.planes[0].buffer.rewind()
        synchronized(frameLock) { full.copyPixelsFromBuffer(image.planes[0].buffer) }
    }

    /**
     * The newest frame, small, for the screen: the scaling is the expensive part and twice
     * a second is plenty to look at.
     */
    private fun keepForScreen(width: Int, height: Int) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastPreviewAt < PREVIEW_INTERVAL_MS) return
        lastPreviewAt = now
        val full = frame ?: return
        val padded = full.width
        val scale = PREVIEW_WIDTH.toFloat() / width
        val visible = synchronized(frameLock) {
            if (padded == width) full else Bitmap.createBitmap(full, 0, 0, width, height)
        }
        val small = Bitmap.createScaledBitmap(visible, PREVIEW_WIDTH, (height * scale).toInt(), true)
        if (visible !== full) visible.recycle()
        // The outgoing preview is not recycled: the screen may still be drawing it, and a
        // recycled bitmap under a Canvas takes the whole app down. It is 360 pixels wide and
        // goes out twice a second; the collector can have it.
        preview = small
        state.value = state.value.copy(preview = small)
    }

    /** The tick: the newest frame through the vision, [TICK_MS] after the last. */
    private val again = Runnable { askForARead() }

    /**
     * Ask the vision for the newest frame, unless it is still busy with the last one.
     *
     * The display hands us a frame only when the screen changes, so a board sitting still
     * produces none and the tick is what keeps the pipeline running over it. That is not a
     * phone quirk to be worked around at some cheaper rate: the desktop's loop runs at ten
     * frames a second and everything downstream counts in those frames -- the tracker's
     * three misses before it drops a lock and its re-fit every ten, the reconciler's memory
     * of a covered cell, how long a cell may stay unexplained before a resync. Reading a
     * couple of times a second stretched every one of them sevenfold, and a board that had
     * locked a few pixels out stayed that way through four moves. So the phone ticks at the
     * rate those constants were written for. What it does not do is the expensive part: the
     * change gate sits on the board's own region and a tick that finds nothing new costs a
     * handful of milliseconds.
     */
    private fun askForARead() {
        handler?.removeCallbacks(again)
        handler?.postDelayed(again, TICK_MS)
        val now = SystemClock.elapsedRealtime()
        if (now - lastReadAt < TICK_MS) return
        // Our own screens are not a game: see SidecoachApp.inFront.
        if (SidecoachApp.inFront) return
        synchronized(frameLock) {
            if (reading) return
            reading = true
        }
        lastReadAt = now
        visionHandler?.post { readTheBoard() } ?: run { reading = false }
    }

    /** The newest frame through the vision, on the vision's own thread. */
    private fun readTheBoard() {
        try {
            val copy = synchronized(frameLock) {
                val full = frame ?: return
                // A copy, so the capture thread may fill the next frame while this one is
                // read: the read is the slow part, and it must not hold the capture up.
                full.copy(Bitmap.Config.ARGB_8888, false)
            } ?: return
            val vision = vision ?: return
            val ts = SystemClock.elapsedRealtime() / 1000.0
            val outcome = runCatching { vision.read(copy, ts) }
                .onFailure { Log.w(TAG, "the frame could not be read: $it") }
                .getOrNull()
            copy.recycle()
            if (outcome == null) return
            state.value = state.value.copy(reading = outcome)
            PanelEngine.watched(
                boardFound = outcome.boardFound,
                verified = outcome.verified,
                fit = outcome.fit,
                fen = outcome.fen,
                ply = outcome.ply,
                userSide = outcome.userSide,
                bottom = outcome.userSide,
                screen = outcome.state,
                following = outcome.following,
                engineOn = outcome.engineOn,
                coachOn = false,
                fps = state.value.latest?.fps ?: 0.0,
                pipelineMs = outcome.detectMs to outcome.readMs,
                coachPending = 0,
                rssMb = null,
            )
            val now = SystemClock.elapsedRealtime()
            if (now - lastOutcomeAt > 5000) {
                lastOutcomeAt = now
                Log.i(
                    TAG,
                    "frame read: board=${outcome.boardFound} at=${outcome.region?.x},${outcome.region?.y} " +
                        "cell=${"%.0f".format(outcome.cellPx)} fit=${"%.2f".format(outcome.fit)} " +
                        "ply=${outcome.ply} ${outcome.state}",
                )
            }
            val fen = outcome.fen
            if (fen != null && fen != lastPlacement) {
                lastPlacement = fen
                Log.i(
                    TAG,
                    "position: $fen (${outcome.unsure} unsure, ${outcome.fresh} embedded, " +
                        "${outcome.readMs} ms)",
                )
            }
        } finally {
            synchronized(frameLock) { reading = false }
        }
    }

    override fun onDestroy() {
        runCatching { display?.release() }
        runCatching { reader?.close() }
        runCatching { projection?.stop() }
        handler?.removeCallbacks(again)
        thread?.quitSafely()
        display = null
        reader = null
        projection = null
        thread = null
        handler = null
        buffer = null
        visionThread?.quitSafely()
        visionThread = null
        visionHandler = null
        runCatching { vision?.close() }
        vision = null
        frame?.recycle()
        frame = null
        preview = null
        state.value = state.value.copy(running = false, preview = null)
        PanelEngine.running(false)
        super.onDestroy()
    }

    private fun notification(): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        if (manager.getNotificationChannel(CHANNEL) == null) {
            manager.createNotificationChannel(
                NotificationChannel(CHANNEL, "Screen capture", NotificationManager.IMPORTANCE_LOW)
                    .also { it.description = "Shown while Sidecoach is watching the screen." },
            )
        }
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, CaptureService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        return NotificationCompat.Builder(this, CHANNEL)
            .setContentTitle("Watching the screen")
            .setContentText("Sidecoach is reading the board.")
            .setSmallIcon(R.drawable.ic_stat_capture)
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(0, "Stop", stop)
            .build()
    }

    /** What the screen shows about the capture; the service is the only writer. */
    data class State(
        val running: Boolean = false,
        val latest: CaptureStats.Snapshot? = null,
        val logPath: String? = null,
        val error: String? = null,
        /** The newest frame, small enough to draw: what the coach is looking at. */
        val preview: Bitmap? = null,
        /** What the vision made of the newest frame it read. */
        val reading: GameWatcher.Watch? = null,
    )

    companion object {
        private const val TAG = "sidecoach.capture"
        private const val CHANNEL = "capture"
        private const val NOTIFICATION_ID = 1
        const val ACTION_STOP = "com.sidecoach.capture.STOP"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"

        /** Half the screen in each direction (a quarter of the pixels). */
        private const val SCALE = 2

        /** The rate the pipeline needs: a board changes when a piece moves, not faster. */
        const val TARGET_FPS = 2
        private const val MIN_INTERVAL_MS = 1000L / TARGET_FPS
        private const val BUFFERS = 3
        private const val PREVIEW_WIDTH = 360
        private const val PREVIEW_INTERVAL_MS = 500L

        /**
         * The desktop's ten frames a second, which is what everything downstream counts in.
         * A read takes as long as it takes and the next is not asked for until it is done,
         * so this is a ceiling, not a promise.
         */
        private const val TICK_MS = 100L

        private val state = MutableStateFlow(State())
        val capture: StateFlow<State> = state.asStateFlow()

        fun start(context: Context, resultCode: Int, data: Intent) {
            val intent = Intent(context, CaptureService::class.java)
                .putExtra(EXTRA_RESULT_CODE, resultCode)
                .putExtra(EXTRA_RESULT_DATA, data)
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            context.startService(
                Intent(context, CaptureService::class.java).setAction(ACTION_STOP),
            )
        }
    }
}
