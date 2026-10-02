package com.sidecoach.vision

import android.util.Log
import java.io.ByteArrayInputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import kotlin.math.sqrt

/**
 * A learned board-and-piece style remembered, so recognition is paid once, not every game:
 * `sidecoach/vision/theme_cache.py`, ported.
 *
 * The recogniser is normally rebuilt from the start position each game: it learns the twelve
 * pieces and the two square colours from the known move-0 layout. That makes
 * reading a puzzle or a mid-game position from scratch impossible -- there is no start
 * position to learn from. This cache persists a learned model keyed by a fingerprint of the
 * board's two square colours, which can be sampled from *any* board without a model. So a
 * theme seen once is recognised forever.
 *
 * The desktop pickles the model. Android has no pickle, so the bytes below are ours: the
 * same payload, written field by field. Everything about the cache's behaviour -- the
 * quantised key, the nearest-palette match within tolerance, a file that does not load
 * being ignored and the theme relearned -- is the Python's, but for the files of older
 * versions, which the phone reads into today's model ([OLDEST_VERSION]).
 */
object ThemeCache {

    /**
     * 2: the landing crops travel with the theme (the desktop's 10); 3: the palette is the
     * desktop's k-means and the start cells' embeddings are held at half precision, as the
     * desktop holds them (the parity audit, 2026-09-26); 4: the embeddings are of the crop at
     * one resolution (`Pieces.canonical`, the desktop's 11); 5: the exemplars keep only what
     * the embedding reads, their tone and brightness, and the landing crops' cell size is
     * always there (the desktop's 12); 6: a landing crop per square colour, each written with
     * its symbol and whether it is on a light square (the desktop's 13). The desktop's
     * numbering is its own file's. Only this version is written.
     */
    const val CACHE_VERSION = 6

    /**
     * The oldest file still read; one older is ignored and the theme relearned. Unlike the
     * desktop's (`theme_cache._read`, which ignores any other version), the phone reads every
     * version from 4 on into today's model: 0.1.19 took the version from 4 to 6 in one release,
     * every learned set on Tai's phone was ignored, and a game joined mid-way said it had not
     * learned the pieces (2026-09-27). What a newer version dropped is read and let go; what
     * it split is shared ([ThemeStore]'s reader says how, version by version). A version
     * raised later must keep reading this one and those after it: `ThemeMigrationTest` holds a
     * file of each, written by that version's own writer.
     */
    const val OLDEST_VERSION = 4

    /** Whether a theme file of [version] is read. */
    fun reads(version: Int): Boolean = version in OLDEST_VERSION..CACHE_VERSION

    /** Square colours within this many levels share a fingerprint (tolerates lighting). */
    private const val QUANTUM = 8

    /** Palettes closer than this (in BGR) are treated as the same theme. */
    const val MATCH_TOLERANCE = 20.0

    /** Whether two palettes are one theme, by the cache's own tolerance (`theme_cache.same_theme`). */
    fun sameTheme(aLight: DoubleArray, aDark: DoubleArray, bLight: DoubleArray, bDark: DoubleArray): Boolean =
        distance(aLight, aDark, bLight, bDark) <= MATCH_TOLERANCE

    private fun distance(
        aLight: DoubleArray, aDark: DoubleArray, bLight: DoubleArray, bDark: DoubleArray,
    ): Double {
        var sum = 0.0
        for (i in 0 until 3) {
            val d = aLight[i] - bLight[i]
            sum += d * d
        }
        for (i in 0 until 3) {
            val d = aDark[i] - bDark[i]
            sum += d * d
        }
        return sqrt(sum)
    }

    /**
     * A stable fingerprint for a board theme from its two square colours. Quantised so
     * near-identical palettes map together, and ordered light-then-dark so the same theme
     * always hashes the same way.
     */
    fun paletteKey(lightBg: DoubleArray, darkBg: DoubleArray): String {
        val quantised = IntArray(6) {
            val v = if (it < 3) lightBg[it] else darkBg[it - 3]
            pyRound(v / QUANTUM)  // `np.round`: halves to even
        }
        // The int64 buffer numpy hashes, little-endian: the desktop's key for the same colours.
        val bytes = ByteArray(6 * 8)
        for (i in 0 until 6) {
            var v = quantised[i].toLong()
            for (b in 0 until 8) { bytes[i * 8 + b] = (v and 0xFF).toByte(); v = v shr 8 }
        }
        val digest = MessageDigest.getInstance("SHA-1").digest(bytes)
        return digest.joinToString("") { "%02x".format(it) }.take(16)
    }

    /**
     * The board's light and dark square colours, sampled from cell corners on any board.
     *
     * The centre of a cell may hold a piece; its corners are almost always the bare square,
     * so a median over every cell's corners reads the palette whether the board is empty or
     * mid-game. Null if the two colours cannot be told apart (a covered board, a bad lock).
     */
    fun samplePalette(frame: org.opencv.core.Mat, geometry: BoardGeometry): Pair<DoubleArray, DoubleArray>? {
        val lights = ArrayList<DoubleArray>()
        val darks = ArrayList<DoubleArray>()
        val inset = maxOf(1, Squares.CELL / 6)
        for (row in 0 until 8) {
            for (col in 0 until 8) {
                val cell = geometry.crop(frame, row, col)
                if (cell.empty() || cell.rows() < 2 * inset || cell.cols() < 2 * inset) {
                    cell.release()
                    continue
                }
                val height = cell.rows()
                val width = cell.cols()
                val bytes = ByteArray(height * width * 3)
                cell.get(0, 0, bytes)
                cell.release()
                val channels = Array(3) { ArrayList<Double>() }
                fun take(y0: Int, y1: Int, x0: Int, x1: Int) {
                    for (y in y0 until y1) {
                        for (x in x0 until x1) {
                            val at = (y * width + x) * 3
                            for (c in 0 until 3) {
                                channels[c].add((bytes[at + c].toInt() and 0xFF).toDouble())
                            }
                        }
                    }
                }
                take(0, inset, 0, inset)
                take(0, inset, width - inset, width)
                take(height - inset, height, 0, inset)
                take(height - inset, height, width - inset, width)
                val corner = DoubleArray(3) { median(channels[it]) }
                if (Squares.isLight(row, col)) lights.add(corner) else darks.add(corner)
            }
        }
        if (lights.isEmpty() || darks.isEmpty()) return null
        val lightBg = DoubleArray(3) { c -> median(lights.map { it[c] }.toMutableList()) }
        val darkBg = DoubleArray(3) { c -> median(darks.map { it[c] }.toMutableList()) }
        var sum = 0.0
        for (c in 0 until 3) { val d = lightBg[c] - darkBg[c]; sum += d * d }
        if (sqrt(sum) < QUANTUM) return null
        return lightBg to darkBg
    }

    private fun median(values: MutableList<Double>): Double {
        if (values.isEmpty()) return 0.0
        values.sort()
        val n = values.size
        return if (n % 2 == 1) values[n / 2] else (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/**
 * A directory of learned themes, one file per board-and-piece set: one board's colours can
 * carry several sets (a set changed, another site with the same colours, a test board), each
 * learned once and kept beside the others; a set learned again replaces itself.
 */
class ThemeStore(private val directory: File) {

    private companion object {
        const val TAG = "sidecoach.themes"
    }

    /**
     * Each file's palette, read once per file version (its size and time, [stamp]): a board
     * waiting to be read cold asks twice a second, and a theme file is a few hundred KB. Only
     * the matching file is loaded. Every version read ([ThemeCache.reads]) begins alike: the
     * version, then the light and the dark square colours.
     */
    private val palettes = HashMap<String, Pair<DoubleArray, DoubleArray>?>()

    private fun path(key: String) = File(directory, "theme-$key.bin")

    private fun stamp(file: File) = "${file.path}:${file.length()}:${file.lastModified()}"

    /**
     * Store a learned model. [key] names the learned set it replaces -- the same pieces on the
     * same board learned again (`GamePipeline.learnedSet`); without one it is kept beside any
     * other set on these board colours. The key, or null.
     */
    fun save(model: ThemeModel, key: String? = null): String? {
        val named = key ?: "${ThemeCache.paletteKey(model.lightBg, model.darkBg)}-${setKey(model)}"
        return store(model, named)
    }

    /**
     * Written beside and moved over the old file: written in place, a save cut short (the app
     * killed, a full disk) left half a file and lost the set it replaced (round 2, finding 9).
     * A save that fails on the disk keeps the old file, says so in the log and returns null;
     * one that fails otherwise raises, to its caller's log. Either way the part file goes
     * (`ThemeCache.save`, round 3).
     */
    private fun store(model: ThemeModel, key: String): String? {
        val target = path(key)
        val partial = File(directory, target.name + ".part")
        var saved = false
        try {
            directory.mkdirs()
            val file = FileOutputStream(partial)
            DataOutputStream(file.buffered()).use { out ->
                write(out, model)
                out.flush()
                file.fd.sync()
            }
            Files.move(partial.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
            saved = true
        } catch (e: IOException) {
            Log.w(TAG, "theme $key not saved: $e")
            return null
        } finally {
            if (!saved) partial.delete()
        }
        return key
    }

    /**
     * Every learned set whose board colours are within tolerance of these, nearest first,
     * with its key. A nearest-palette match rather than an exact key so a shade of rendering
     * noise between sessions never causes a miss; a user has only a handful of themes. Each
     * model is read afresh, so every caller gets one of its own.
     */
    fun candidates(lightBg: DoubleArray, darkBg: DoubleArray): List<Pair<String, ThemeModel>> {
        val files = directory.listFiles { f -> f.name.startsWith("theme-") && f.name.endsWith(".bin") }
            ?.sortedBy { it.name } ?: return emptyList()
        val near = ArrayList<Pair<Double, File>>()
        for (file in files) {
            val palette = palette(file) ?: continue
            val d = distanceTo(lightBg, darkBg, palette)
            if (d <= ThemeCache.MATCH_TOLERANCE) near.add(d to file)
        }
        val found = ArrayList<Pair<String, ThemeModel>>()
        for ((_, file) in near.sortedBy { it.first }) {
            // Any failure to read is a file not there, said once in the log: this version of the
            // file is not read again, as the desktop's palette, read from the whole file, is
            // not. The file is read whole first, so every length in it is held to the bytes left
            // ([length]) before anything is allocated by it: a damaged length asked for more
            // memory than there is (round 2, finding 9; round 3, where an OutOfMemoryError was
            // caught instead).
            val model = try {
                DataInputStream(ByteArrayInputStream(file.readBytes())).use { read(it) }
            } catch (e: Exception) {
                Log.w(TAG, "theme file ${file.name} not read, left out: $e")
                palettes[stamp(file)] = null
                null
            } ?: continue
            found.add(file.name.removePrefix("theme-").removeSuffix(".bin") to model)
        }
        return found
    }

    /** The learned model whose palette is nearest this one within tolerance, or null. */
    fun get(lightBg: DoubleArray, darkBg: DoubleArray): ThemeModel? = candidates(lightBg, darkBg).firstOrNull()?.second

    /**
     * A name for a learned set among those on the same board colours, from its start cells'
     * embeddings: every learned set has them (`theme_cache._set_key`).
     */
    private fun setKey(model: ThemeModel): String {
        // The half-precision buffer the desktop hashes, little-endian.
        val digest = MessageDigest.getInstance("SHA-1")
        val bytes = java.nio.ByteBuffer.allocate(2 * model.exemplarVectors[0].size).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (vector in model.exemplarVectors) {
            bytes.clear()
            for (v in vector) bytes.putShort(halfBits(v))
            digest.update(bytes.array(), 0, bytes.position())
        }
        return digest.digest().joinToString("") { "%02x".format(it) }.take(8)
    }

    private fun distanceTo(
        lightBg: DoubleArray, darkBg: DoubleArray, palette: Pair<DoubleArray, DoubleArray>,
    ): Double {
        var sum = 0.0
        for (i in 0 until 3) { val d = lightBg[i] - palette.first[i]; sum += d * d }
        for (i in 0 until 3) { val d = darkBg[i] - palette.second[i]; sum += d * d }
        return sqrt(sum)
    }

    private fun palette(file: File): Pair<DoubleArray, DoubleArray>? {
        val key = stamp(file)
        if (!palettes.containsKey(key)) {
            palettes[key] = try {
                DataInputStream(file.inputStream().buffered()).use { stream ->
                    if (!ThemeCache.reads(stream.readInt())) return@use null
                    val light = DoubleArray(3) { stream.readDouble() }
                    val dark = DoubleArray(3) { stream.readDouble() }
                    light to dark
                }
            } catch (e: Exception) {
                Log.w(TAG, "theme file ${file.name} not read, left out: $e")
                null
            }
        }
        return palettes[key]
    }

    /**
     * A length read from a theme file, of [count] items of [bytesEach] bytes each, held to the
     * bytes left in it: a damaged one is refused before anything is allocated by it.
     */
    private fun DataInputStream.length(count: Long, bytesEach: Long = 1): Int {
        if (count < 0 || count > available() / bytesEach) {
            throw IOException("damaged theme file: $count items of $bytesEach bytes past its end")
        }
        return count.toInt()
    }

    // The payload, written at CACHE_VERSION only. A file that does not read back, or is older
    // than OLDEST_VERSION, is ignored and the theme relearned.

    private fun write(out: DataOutputStream, model: ThemeModel) {
        out.writeInt(ThemeCache.CACHE_VERSION)
        for (v in model.lightBg) out.writeDouble(v)
        for (v in model.darkBg) out.writeDouble(v)
        out.writeBoolean(model.whiteAtBottom)
        out.writeInt(model.exemplars.size)
        for (exemplar in model.exemplars) {
            out.writeUTF(exemplar.symbol)
            writeBits(out, exemplar.tone)
            out.writeDouble(exemplar.brightness)
        }
        out.writeInt(model.pieceColours.size)
        for ((fill, edge) in model.pieceColours) {
            for (v in fill) out.writeDouble(v)
            out.writeBoolean(edge != null)
            if (edge != null) for (v in edge) out.writeDouble(v)
        }
        out.writeInt(model.palette.size)
        for (colour in model.palette) for (v in colour) out.writeDouble(v)
        out.writeInt(model.exemplarVectors.size)
        out.writeInt(if (model.exemplarVectors.isEmpty()) 0 else model.exemplarVectors[0].size)
        // Half precision, as the desktop pickles them: the model holds them at it already.
        for (vector in model.exemplarVectors) for (v in vector) out.writeShort(halfBits(v).toInt())
        out.writeInt(model.exemplarLabels.size)
        for (label in model.exemplarLabels) out.writeUTF(label)
        out.writeInt(model.rankLabels.size)
        for ((rank, glyph) in model.rankLabels) {
            out.writeInt(rank)
            writeBits(out, glyph)
        }
        out.writeInt(model.landingCrops.size)
        for ((key, crop) in model.landingCrops) {
            out.writeUTF(key.first)
            out.writeBoolean(key.second)
            out.writeInt(crop.rows)
            out.writeInt(crop.cols)
            out.write(crop.pixels)
        }
        out.writeDouble(model.landingPx)
    }

    /**
     * A theme file of any version [ThemeCache.reads], as today's model. The versions differ in
     * two places, and the rest of the file is the same in all of them:
     *
     * - An exemplar of version 4 also carries its silhouette mask, after its symbol, and three
     *   centroids (the silhouette's, the one it was aligned by, the one it landed by: two
     *   doubles each), after its brightness; 5 dropped them with the matcher that used them
     *   (desktop e0ea564). They are read and let go.
     * - A landing crop of version 4 or 5 is one per piece kind, the one 0.1.18 measured a piece
     *   against on either square colour; 6 keeps one per kind and square colour (desktop
     *   1eab427). An old crop stands for both colours, as it did. And version 4 wrote the cell
     *   size the crops were taken at only from desktop cfaa416 on: a file without it compares
     *   at full detail (0), as it did.
     */
    private fun read(input: DataInputStream): ThemeModel? {
        val version = input.readInt()
        if (!ThemeCache.reads(version)) return null
        val lightBg = DoubleArray(3) { input.readDouble() }
        val darkBg = DoubleArray(3) { input.readDouble() }
        val whiteAtBottom = input.readBoolean()
        val exemplars = ArrayList<Exemplar>()
        repeat(input.readInt()) {
            val symbol = input.readUTF()
            if (version <= 4) readBits(input) // the silhouette mask
            val tone = readBits(input)
            exemplars.add(Exemplar(symbol, tone, input.readDouble()))
            if (version <= 4) repeat(6) { input.readDouble() } // the three centroids
        }
        val pieceColours = ArrayList<Pair<DoubleArray, DoubleArray?>>()
        repeat(input.readInt()) {
            val fill = DoubleArray(3) { input.readDouble() }
            val edge = if (input.readBoolean()) DoubleArray(3) { input.readDouble() } else null
            pieceColours.add(fill to edge)
        }
        val palette = ArrayList<DoubleArray>()
        repeat(input.readInt()) { palette.add(DoubleArray(3) { input.readDouble() }) }
        val vectorCount = input.readInt()
        val dim = input.readInt()
        if (dim < 0) throw IOException("damaged theme file: vectors of $dim numbers")
        input.length(vectorCount.toLong() * dim, 2) // half precision: two bytes a number
        val vectors = Array(vectorCount) { FloatArray(dim) { halfToFloat(input.readShort()) } }
        val labels = ArrayList<String>()
        repeat(input.readInt()) { labels.add(input.readUTF()) }
        val rankLabels = LinkedHashMap<Int, BooleanArray>()
        repeat(input.readInt()) {
            val rank = input.readInt()
            rankLabels[rank] = readBits(input)
        }
        val landingCrops = LinkedHashMap<Pair<String, Boolean>, LandingCrop>()
        repeat(input.readInt()) {
            val symbol = input.readUTF()
            val light = if (version >= 6) input.readBoolean() else null
            val rows = input.readInt()
            val cols = input.readInt()
            if (rows < 0 || cols < 0) throw IOException("damaged theme file: a crop of $rows x $cols")
            val pixels = ByteArray(input.length(rows.toLong() * cols))
            input.readFully(pixels)
            val crop = LandingCrop(rows, cols, pixels)
            if (light != null) {
                landingCrops[symbol to light] = crop
            } else {
                landingCrops[symbol to true] = crop
                landingCrops[symbol to false] = crop
            }
        }
        val landingPx = if (version >= 5) {
            input.readDouble()
        } else {
            try {
                input.readDouble()
            } catch (e: java.io.EOFException) {
                0.0 // cached before the size was kept
            }
        }
        val model = ThemeModel(
            exemplars = exemplars,
            lightBg = lightBg,
            darkBg = darkBg,
            pieceColours = pieceColours,
            palette = palette,
            exemplarVectors = vectors,
            exemplarLabels = labels,
            whiteAtBottom = whiteAtBottom,
            landingCrops = landingCrops,
            landingPx = landingPx,
        )
        model.rankLabels = rankLabels
        return model
    }

    private fun writeBits(out: DataOutputStream, bits: BooleanArray) {
        out.writeInt(bits.size)
        var byte = 0
        for (i in bits.indices) {
            if (bits[i]) byte = byte or (1 shl (i % 8))
            if (i % 8 == 7) { out.writeByte(byte); byte = 0 }
        }
        if (bits.size % 8 != 0) out.writeByte(byte)
    }

    private fun readBits(input: DataInputStream): BooleanArray {
        val size = input.readInt()
        if (size < 0) throw IOException("damaged theme file: $size bits")
        input.length((size + 7L) / 8) // eight bits a byte
        val bits = BooleanArray(size)
        var byte = 0
        for (i in 0 until size) {
            if (i % 8 == 0) byte = input.readUnsignedByte()
            bits[i] = (byte shr (i % 8)) and 1 == 1
        }
        return bits
    }
}
