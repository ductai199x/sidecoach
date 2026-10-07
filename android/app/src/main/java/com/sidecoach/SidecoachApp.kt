package com.sidecoach

import android.app.Activity
import android.app.Application
import android.os.Bundle
import com.sidecoach.coach.Corrections

/** A screen of the app's that is not Sidecoach in front: what it shows is read as a game. */
interface NotInFront

/**
 * The app, and the one thing it needs to know process-wide: whether one of its own screens
 * is what the display is showing.
 *
 * The capture mirrors the whole display and cannot leave a window out, so while Sidecoach
 * is in front the frames are of Sidecoach -- its preview inside its preview -- and the
 * detector locks onto the board drawn in that preview and reads it at the wrong scale.
 * Nothing is read then (Tai, 2026-09-22): the person is looking at the coach, not at a
 * game. The desktop does the same thing by masking its own window and vetoing its own
 * board by signature (vision/ownboard.py).
 *
 * A [NotInFront] screen is the exception: the debug build's test board (debug.BoardActivity,
 * src/debug), whose whole purpose is to put a rendered board on the display for the vision
 * to read.
 */
class SidecoachApp : Application() {

    override fun onCreate() {
        super.onCreate()
        // C, the lessons at the end of the coach's system message, read once before any coach
        // is built (a coach writes its system message when it is made). The desktop reads its
        // file on first use (`prompt.corrections`); nothing on the phone read the asset, and
        // every model call went without the lessons (the parity audit cc-01, 2026-09-26).
        Corrections.load(assets)
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityResumed(activity: Activity) {
                if (activity !is NotInFront) showing++
            }

            override fun onActivityPaused(activity: Activity) {
                if (activity !is NotInFront) showing--
            }

            override fun onActivityCreated(activity: Activity, state: Bundle?) = Unit
            override fun onActivityStarted(activity: Activity) = Unit
            override fun onActivityStopped(activity: Activity) = Unit
            override fun onActivitySaveInstanceState(activity: Activity, out: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = Unit
        })
    }

    companion object {
        @Volatile
        private var showing: Int = 0

        /** Whether one of Sidecoach's own screens is in front of the person right now. */
        val inFront: Boolean get() = showing > 0
    }
}
