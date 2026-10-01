package com.sidecoach.vision

import java.security.MessageDigest

/**
 * A 16-byte digest of a crop's pixels and what they were read with: the key the vision's
 * caches keep their results by (`hashlib.blake2b(digest_size=16)` in `pieces.silhouette` and
 * `CellEmbedder.embed`, desktop a3479bd). MD5 from the platform's native provider: a crop's
 * bytes go in one call.
 */
internal class Digest private constructor(private val high: Long, private val low: Long) {

    override fun equals(other: Any?): Boolean = other is Digest && other.high == high && other.low == low

    override fun hashCode(): Int = (low xor (low ushr 32)).toInt()

    /** What is fed after the bytes: numbers as their bits, gathered and given in one call. */
    class Feed internal constructor() {
        internal var bytes = ByteArray(256)
        internal var count = 0

        fun longs(vararg values: Long): Feed {
            for (v in values) long(v)
            return this
        }

        fun doubles(values: DoubleArray): Feed {
            for (v in values) long(java.lang.Double.doubleToLongBits(v))
            return this
        }

        private fun long(value: Long) {
            if (count + 8 > bytes.size) bytes = bytes.copyOf(bytes.size * 2)
            var v = value
            for (b in 0 until 8) {
                bytes[count++] = (v and 0xFF).toByte()
                v = v ushr 8
            }
        }
    }

    companion object {
        private val digests: ThreadLocal<MessageDigest> = ThreadLocal.withInitial { MessageDigest.getInstance("MD5") }

        /** The digest of [bytes] and whatever [more] feeds after them. */
        fun of(bytes: ByteArray, more: (Feed) -> Unit = {}): Digest {
            val digest = digests.get()!!
            digest.reset()
            digest.update(bytes)
            val feed = Feed()
            more(feed)
            if (feed.count > 0) digest.update(feed.bytes, 0, feed.count)
            val out = digest.digest()
            var high = 0L
            var low = 0L
            for (i in 0 until 8) high = (high shl 8) or (out[i].toLong() and 0xFF)
            for (i in 8 until 16) low = (low shl 8) or (out[i].toLong() and 0xFF)
            return Digest(high, low)
        }
    }
}
