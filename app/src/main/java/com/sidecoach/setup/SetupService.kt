package com.sidecoach.setup

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import com.sidecoach.MainActivity
import com.sidecoach.R
import org.json.JSONObject

/**
 * What the download's notification shows, from the setup's run state ([Setup.watcher]): the
 * item under way and how far the run is, in whole percent; nothing once the run is over. Each
 * state is handed to [show] only when what is shown changes (a percent more, the next item, the
 * end), not at every chunk. The phone's own (Tai, 2026-10-02): Android may stop an app in the
 * background, and the setup's downloads, on a plain thread, went with it.
 */
class SetupNotice(private val show: (Shown?) -> Unit) {

    /** The item under way (its title), and the run's whole percent, null while unknown. */
    data class Shown(val title: String, val percent: Int?)

    private var last: Shown? = null

    @Synchronized
    fun on(state: JSONObject) {
        val next = if (!state.optBoolean("running")) null else shown(state)
        if (next == last) return
        last = next
        show(next)
    }

    private fun shown(state: JSONObject): Shown {
        val item = state.optString("item").takeIf { !state.isNull("item") && it.isNotEmpty() }
        val title = item?.let { state.optJSONObject("titles")?.optString(it)?.ifEmpty { null } }
            ?: "Getting ready"
        val total = state.optLong("total")
        val percent = if (total > 0) (state.optLong("done") * 100 / total).toInt().coerceIn(0, 100) else null
        return Shown(title, percent)
    }
}

/**
 * The setup's downloads held in the foreground while they run (Tai, 2026-10-02): Android stops
 * an app in the background as it needs, and a foreground service is how it lets a download go
 * on; it asks a notification of it, so the notification is there while a download runs and
 * gone the moment it ends. Progress and Cancel stay the page's, as before; the notification
 * shows the same progress and opens the app. Started and stopped by [follow].
 */
class SetupService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // A start by startForegroundService owes a startForeground, even when the run ended
        // before the service came up: then it goes at once.
        val shown = current
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification(this, shown ?: READY), TYPE)
        if (shown == null) {
            end()
        } else {
            running = this
            if (current == null) end() // ended meanwhile
        }
        return START_NOT_STICKY
    }

    /** Android's limit on a data-sync service's time reached: the download goes on unheld. */
    override fun onTimeout(startId: Int, fgsType: Int) {
        end()
    }

    override fun onDestroy() {
        if (running === this) running = null
        super.onDestroy()
    }

    private fun end() {
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    companion object {
        private const val CHANNEL = "downloads"
        private const val NOTIFICATION_ID = 2
        private const val TYPE = ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        private val READY = SetupNotice.Shown("Getting ready", null)

        @Volatile private var current: SetupNotice.Shown? = null

        /** The service once it is in the foreground: the run's end stops it from here. */
        @Volatile private var running: SetupService? = null

        /** The setup's runs held in the foreground: its [Setup.watcher], through a [SetupNotice]. */
        fun follow(context: Context, setup: Setup) {
            val app = context.applicationContext
            val notice = SetupNotice { shown -> show(app, shown) }
            setup.watcher = notice::on
        }

        private fun show(context: Context, shown: SetupNotice.Shown?) {
            val was = current
            current = shown
            when {
                // Never stopped from outside: a service started for the foreground and stopped
                // before its startForeground is a crash. It ends itself, now or as it comes up.
                shown == null -> running?.let { service -> service.mainExecutor.execute { service.end() } }
                was == null -> runCatching {
                    context.startForegroundService(Intent(context, SetupService::class.java))
                }
                else -> context.getSystemService(NotificationManager::class.java)
                    .notify(NOTIFICATION_ID, notification(context, shown))
            }
        }

        private fun notification(context: Context, shown: SetupNotice.Shown): Notification {
            val manager = context.getSystemService(NotificationManager::class.java)
            if (manager.getNotificationChannel(CHANNEL) == null) {
                manager.createNotificationChannel(
                    NotificationChannel(CHANNEL, "Downloads", NotificationManager.IMPORTANCE_LOW)
                        .also { it.description = "Shown while Sidecoach downloads what it needs." },
                )
            }
            val open = PendingIntent.getActivity(
                context,
                0,
                Intent(context, MainActivity::class.java),
                PendingIntent.FLAG_IMMUTABLE,
            )
            return NotificationCompat.Builder(context, CHANNEL)
                .setContentTitle("Downloading")
                .setContentText(shown.title)
                .setSmallIcon(R.drawable.ic_stat_capture)
                .setColor(context.getColor(R.color.amber))
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setProgress(100, shown.percent ?: 0, shown.percent == null)
                .setContentIntent(open)
                .build()
        }
    }
}
