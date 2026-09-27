package com.sidecoach.voice

/**
 * What speaks the coach's lines: `sidecoach/voice/base.py`'s `TTSProvider`. [KokoroTts] is the
 * phone's; the panel speaks through this and nothing more, so a test can hand it its own, as the
 * desktop's tests hand the panel theirs.
 *
 * Speaking never takes the session down: a provider that cannot load its model says so from
 * [healthcheck] and turns [say] into a no-op.
 */
interface TtsProvider {
    val name: String

    /** (ready, detail): whether the voice can speak, and why not when it cannot. */
    fun healthcheck(): Pair<Boolean, String>

    /**
     * Queue a line; chess notation in it is read aloud in words. Returns once it is queued.
     * [voice] speaks this one line in another voice (the settings page's preview).
     */
    fun say(text: String, voice: String? = null)

    /** Cut off anything now playing or queued: a barge-in, a new position. */
    fun stop()

    /** Whether a line is playing or waiting to play. */
    val busy: Boolean
}
