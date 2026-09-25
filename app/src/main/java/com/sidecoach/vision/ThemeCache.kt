package com.sidecoach.vision

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.security.MessageDigest
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * A learned board-and-piece style remembered, so recognition is paid once, not every game:
 * `sidecoach/vision/theme_cache.py`, ported.
 *
 * The recogniser is normally rebuilt from the start position each game: it learns the twelve
 * piece silhouettes and the two square colours from the known move-0 layout. That makes
 * reading a puzzle or a mid-game position from scratch impossible -- there is no start
 * position to learn from. This cache persists a learned model keyed by a fingerprint of the
 * board's two square colours, which can be sampled from *any* board without a model. So a
 * theme seen once is recognised forever.
 *
 * The desktop pickles the model. Android has no pickle, so the bytes below are ours: the
 * same payload, written field by field. Everything about the cache's behaviour -- the
 * quantised key, the nearest-palette match within tolerance, a file that no longer loads
 * being ignored and the theme relearned -- is the Python's.
 */
object ThemeCache {

    /**
     * 2: the landing crops travel with the theme (the desktop's 10). 1, the first Android
     * format, has none and still reads: its pieces land by the centroid until the set is
     * learned again from a start position. The desktop's numbering is its own file's.
     */
    const val CACHE_VERSION = 2
    const val CACHE_VERSION_NO_CROPS = 1

    /** Square colours within this many levels share a fingerprint (tolerates lighting). */
    private const val QUANTUM = 8

    /** Palettes closer than this (in BGR) are treated as the same theme. */
    const val MATCH_TOLERANCE = 20.0

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
            (v / QUANTUM).roundToInt()
        }
        // numpy hashes the int64 buffer; the bytes here are ours, and only have to be stable.
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

    /**
     * Each file's palette, read once per file version: a board waiting to be read cold asks
     * twice a second, and a theme file is a few hundred KB. Only the matching file is loaded.
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

    private fun store(model: ThemeModel, key: String): String? {
        directory.mkdirs()
        try {
            DataOutputStream(path(key).outputStream().buffered()).use { write(it, model) }
        } catch (e: Exception) {
            return null
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
            val model = try {
                DataInputStream(file.inputStream().buffered()).use { read(it) }
            } catch (e: Exception) {
                null
            } ?: continue
            found.add(file.name.removePrefix("theme-").removeSuffix(".bin") to model)
        }
        return found
    }

    /** The learned model whose palette is nearest this one within tolerance, or null. */
    fun get(lightBg: DoubleArray, darkBg: DoubleArray): ThemeModel? = candidates(lightBg, darkBg).firstOrNull()?.second

    /**
     * A name for a learned set among those on the same board colours: from its start cells'
     * embeddings, or a fresh one when it has none.
     */
    private fun setKey(model: ThemeModel): String {
        if (model.exemplarVectors.isEmpty()) return java.util.UUID.randomUUID().toString().replace("-", "").take(8)
        val digest = MessageDigest.getInstance("SHA-1")
        val bytes = java.nio.ByteBuffer.allocate(4 * model.exemplarVectors[0].size).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        for (vector in model.exemplarVectors) {
            bytes.clear()
            for (v in vector) bytes.putFloat(v)
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
                    val version = stream.readInt()
                    if (version != ThemeCache.CACHE_VERSION && version != ThemeCache.CACHE_VERSION_NO_CROPS) {
                        return@use null
                    }
                    val light = DoubleArray(3) { stream.readDouble() }
                    val dark = DoubleArray(3) { stream.readDouble() }
                    light to dark
                }
            } catch (e: Exception) {
                null
            }
        }
        return palettes[key]
    }

    // The payload. A file that does not read back is ignored and the theme relearned.

    private fun write(out: DataOutputStream, model: ThemeModel) {
        out.writeInt(ThemeCache.CACHE_VERSION)
        for (v in model.lightBg) out.writeDouble(v)
        for (v in model.darkBg) out.writeDouble(v)
        out.writeBoolean(model.whiteAtBottom)
        out.writeInt(model.exemplars.size)
        for (exemplar in model.exemplars) {
            out.writeUTF(exemplar.symbol)
            writeBits(out, exemplar.mask)
            writeBits(out, exemplar.tone)
            out.writeDouble(exemplar.brightness)
            out.writeDouble(exemplar.centroid.first); out.writeDouble(exemplar.centroid.second)
            out.writeDouble(exemplar.alignCentroid.first); out.writeDouble(exemplar.alignCentroid.second)
            out.writeDouble(exemplar.landingCentroid.first); out.writeDouble(exemplar.landingCentroid.second)
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
        for (vector in model.exemplarVectors) for (v in vector) out.writeFloat(v)
        out.writeInt(model.exemplarLabels.size)
        for (label in model.exemplarLabels) out.writeUTF(label)
        out.writeInt(model.rankLabels.size)
        for ((rank, glyph) in model.rankLabels) {
            out.writeInt(rank)
            writeBits(out, glyph)
        }
        out.writeInt(model.landingCrops.size)
        for ((symbol, crop) in model.landingCrops) {
            out.writeUTF(symbol)
            out.writeInt(crop.rows)
            out.writeInt(crop.cols)
            out.write(crop.pixels)
        }
    }

    private fun read(input: DataInputStream): ThemeModel? {
        val version = input.readInt()
        if (version != ThemeCache.CACHE_VERSION && version != ThemeCache.CACHE_VERSION_NO_CROPS) return null
        val lightBg = DoubleArray(3) { input.readDouble() }
        val darkBg = DoubleArray(3) { input.readDouble() }
        val whiteAtBottom = input.readBoolean()
        val exemplars = ArrayList<Exemplar>()
        repeat(input.readInt()) {
            val symbol = input.readUTF()
            val mask = readBits(input)
            val tone = readBits(input)
            val brightness = input.readDouble()
            val centroid = input.readDouble() to input.readDouble()
            val align = input.readDouble() to input.readDouble()
            val landing = input.readDouble() to input.readDouble()
            exemplars.add(Exemplar(symbol, mask, tone, brightness, centroid, align, landing))
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
        val vectors = Array(vectorCount) { FloatArray(dim) { input.readFloat() } }
        val labels = ArrayList<String>()
        repeat(input.readInt()) { labels.add(input.readUTF()) }
        val rankLabels = LinkedHashMap<Int, BooleanArray>()
        repeat(input.readInt()) {
            val rank = input.readInt()
            rankLabels[rank] = readBits(input)
        }
        val landingCrops = LinkedHashMap<String, LandingCrop>()
        if (version == ThemeCache.CACHE_VERSION) {
            repeat(input.readInt()) {
                val symbol = input.readUTF()
                val rows = input.readInt()
                val cols = input.readInt()
                val pixels = ByteArray(rows * cols)
                input.readFully(pixels)
                landingCrops[symbol] = LandingCrop(rows, cols, pixels)
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
        val bits = BooleanArray(size)
        var byte = 0
        for (i in 0 until size) {
            if (i % 8 == 0) byte = input.readUnsignedByte()
            bits[i] = (byte shr (i % 8)) and 1 == 1
        }
        return bits
    }
}
