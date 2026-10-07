package com.sidecoach.voice

/**
 * Who gave a line ([TtsProvider.say]'s owner): the live coach; the rooms name their own (a
 * review, the puzzles), and a room's stop stops only its own lines (round 3 audit, E5).
 * `voice.base.COACH`.
 */
const val COACH = "coach"

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
     * [voice] speaks this one line in another voice (the settings page's preview). [owner] is
     * who gave the line: the coach, or a room (a review, the puzzles).
     */
    fun say(text: String, voice: String? = null, owner: String = COACH)

    /**
     * Cut off anything now playing or queued: a barge-in, a new position. With [owner], only
     * that owner's lines, queued or playing (a room's stop).
     */
    fun stop(owner: String? = null)

    /** Whether a line is playing or waiting to play. */
    val busy: Boolean

    /**
     * How many lines wait behind the one playing (the next, made meanwhile, counted); null when
     * the voice cannot say -- then the pace plan hands it a line only when it is idle. The
     * desktop's optional `waiting`.
     */
    val waiting: Int? get() = null

    /**
     * Drop [owner]'s lines waiting behind the one playing, the one made ahead too, and leave the
     * one playing to finish (the pace plan: what waits is no longer the news). `drop_waiting`.
     */
    fun dropWaiting(owner: String = COACH) {}

    /** Make [text] now, to play at once when it is said (a fast line's word). `prepare`. */
    fun prepare(text: String) {}
}
