package com.sidecoach.capture

import android.content.Context
import android.os.BatteryManager
import android.os.PowerManager
import android.os.SystemClock
import org.json.JSONObject
import java.io.File

/**
 * What Spike A is for: whether the phone hands us frames steadily, and what that costs.
 *
 * Every frame's interval and the time to copy it out of the [android.media.ImageReader]
 * are recorded; every [WINDOW_MS] a snapshot of those, of this process's CPU, of the
 * battery and of the thermal status goes to a JSONL file and to the screen. Nothing here
 * keeps a frame: the pixels are measured and dropped.
 */
class CaptureStats(private val context: Context, private val log: File) {

    /** One window's worth, as the screen shows it and the file keeps it. */
    data class Snapshot(
        val elapsedS: Double = 0.0,
        val frames: Int = 0,
        val fps: Double = 0.0,
        val dropped: Int = 0,
        val intervalP50: Int = 0,
        val intervalP95: Int = 0,
        val intervalMax: Int = 0,
        val copyP50: Int = 0,
        val copyP95: Int = 0,
        val frameKb: Int = 0,
        val cpuPercent: Double = 0.0,
        val batteryPercent: Int = -1,
        val batteryMa: Int = 0,
        val thermal: String = "?",
    )

    private val intervals = ArrayList<Int>(256)
    private val copies = ArrayList<Int>(256)
    private var frames = 0
    private var dropped = 0
    private var frameBytes = 0
    private var lastFrameAt = 0L
    private var windowBegan = SystemClock.elapsedRealtime()
    private val began = SystemClock.elapsedRealtime()
    private var cpuAtWindow = processCpuMs()

    /** A frame arrived and was copied out in [copyMs]; [bytes] is what it measured. */
    @Synchronized
    fun frame(copyMs: Int, bytes: Int) {
        val now = SystemClock.elapsedRealtime()
        if (lastFrameAt != 0L) intervals.add((now - lastFrameAt).toInt())
        lastFrameAt = now
        copies.add(copyMs)
        frameBytes = bytes
        frames++
    }

    /** A frame the reader had ready and the throttle skipped. */
    @Synchronized
    fun skipped() {
        dropped++
    }

    /** True once a window is up and [take] is worth calling. */
    fun due(): Boolean = SystemClock.elapsedRealtime() - windowBegan >= WINDOW_MS

    /** The window's snapshot; the counters start again. Appends one line to the log. */
    @Synchronized
    fun take(): Snapshot {
        val now = SystemClock.elapsedRealtime()
        val windowS = (now - windowBegan) / 1000.0
        val cpuNow = processCpuMs()
        val snapshot = Snapshot(
            elapsedS = (now - began) / 1000.0,
            frames = frames,
            fps = if (windowS > 0) frames / windowS else 0.0,
            dropped = dropped,
            intervalP50 = percentile(intervals, 50),
            intervalP95 = percentile(intervals, 95),
            intervalMax = intervals.maxOrNull() ?: 0,
            copyP50 = percentile(copies, 50),
            copyP95 = percentile(copies, 95),
            frameKb = frameBytes / 1024,
            // Of one core: 100% means this process kept a core busy for the window.
            cpuPercent = if (windowS > 0) (cpuNow - cpuAtWindow) / (windowS * 10.0) else 0.0,
            batteryPercent = batteryPercent(),
            batteryMa = batteryMilliamps(),
            thermal = thermalWord(),
        )
        intervals.clear()
        copies.clear()
        frames = 0
        dropped = 0
        windowBegan = now
        cpuAtWindow = cpuNow
        append(snapshot)
        return snapshot
    }

    private fun append(s: Snapshot) {
        val line = JSONObject()
            .put("t", s.elapsedS)
            .put("frames", s.frames)
            .put("fps", round2(s.fps))
            .put("dropped", s.dropped)
            .put("interval_p50", s.intervalP50)
            .put("interval_p95", s.intervalP95)
            .put("interval_max", s.intervalMax)
            .put("copy_p50", s.copyP50)
            .put("copy_p95", s.copyP95)
            .put("frame_kb", s.frameKb)
            .put("cpu_percent", round2(s.cpuPercent))
            .put("battery_percent", s.batteryPercent)
            .put("battery_ma", s.batteryMa)
            .put("thermal", s.thermal)
            .toString()
        runCatching {
            log.parentFile?.mkdirs()
            log.appendText(line + "\n")
        }
    }

    private fun batteryPercent(): Int =
        runCatching {
            val manager = context.getSystemService(BatteryManager::class.java)
            manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        }.getOrDefault(-1)

    /** Negative while the battery is being drained; the sign is the device's own. */
    private fun batteryMilliamps(): Int =
        runCatching {
            val manager = context.getSystemService(BatteryManager::class.java)
            manager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW) / 1000
        }.getOrDefault(0)

    private fun thermalWord(): String =
        runCatching {
            when (context.getSystemService(PowerManager::class.java).currentThermalStatus) {
                PowerManager.THERMAL_STATUS_NONE -> "none"
                PowerManager.THERMAL_STATUS_LIGHT -> "light"
                PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
                PowerManager.THERMAL_STATUS_SEVERE -> "severe"
                PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
                PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
                PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
                else -> "?"
            }
        }.getOrDefault("?")

    /**
     * This process's CPU time in milliseconds: fields 14 and 15 of /proc/self/stat, in
     * clock ticks. The tick is 100 Hz on every Android device shipped so far; the number
     * is for comparing runs, not for a bill.
     */
    private fun processCpuMs(): Long =
        runCatching {
            val stat = File("/proc/self/stat").readText()
            // The second field is the command, in brackets, and may hold spaces.
            val after = stat.substring(stat.lastIndexOf(')') + 2).split(' ')
            val utime = after[11].toLong()
            val stime = after[12].toLong()
            (utime + stime) * 10
        }.getOrDefault(0L)

    private fun percentile(values: List<Int>, p: Int): Int {
        if (values.isEmpty()) return 0
        val sorted = values.sorted()
        val index = (sorted.size - 1) * p / 100
        return sorted[index]
    }

    private fun round2(v: Double): Double = Math.round(v * 100.0) / 100.0

    companion object {
        const val WINDOW_MS = 10_000L
    }
}
