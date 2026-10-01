package com.sidecoach.ui

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Base64

/**
 * How the boards look (`sidecoach/looks.py`): the piece sets Sidecoach ships, the board colour
 * themes, and a player's own piece sets.
 *
 * The shipped sets are Sidecoach's own (the default) and every Lichess set whose licence allows
 * any use: static files beside the page, under /pieces/ and /pieces/<set>/. The themes are
 * plain colours. A player's own sets are images they choose, each set named and kept apart in
 * the app's files (pieces/own/<id>/, its name in set.json) and handed to the page inline. A set
 * deleted waits [UNDO_S] in pieces/own/.removed/ for its Undo. The one own set there was before
 * (pieces/custom) becomes the set "My pieces" (own:custom). Nothing names or fetches any other
 * product's set.
 *
 * The board's squares carry the panel's signature for the desktop's vision: its odd rows three
 * units bluer than its even rows, square colour and highlight alike; every theme keeps it.
 */
object Looks {
    const val SHADE_STEP = 3
    val CODES: List<String> = "wb".flatMap { c -> "KQRBNP".map { p -> "$c$p" } }
    const val DEFAULT_SET = "sidecoach"
    const val CUSTOM = "custom" // the one own set before there were several; now the own set of that id
    const val OWN = "own:" // an own set as the board setting names it: "own:<id>"
    const val DEFAULT_THEME = "green"

    class PieceSet(val label: String, val author: String, val licence: String, val folder: String, val version: String) {
        fun url(code: String): String = "/pieces/$folder$code.svg?v=$version"
    }

    val PIECE_SETS: Map<String, PieceSet> = linkedMapOf(
        "sidecoach" to PieceSet("Sidecoach", "Sidecoach", "Sidecoach's own", "", "own2"),
        "cburnett" to PieceSet("Classic", "Colin M.L. Burnett", "GPLv2+", "cburnett/", "1"),
        "merida" to PieceSet("Merida", "Armando Hernandez Marroquin", "GPLv2+", "merida/", "1"),
        "chessnut" to PieceSet("Chessnut", "Alexis Luengas", "Apache 2.0", "chessnut/", "1"),
        "celtic" to PieceSet("Celtic", "Maurizio Monge", "MIT", "celtic/", "1"),
        "spatial" to PieceSet("Spatial", "Maurizio Monge", "MIT", "spatial/", "1"),
        "fantasy" to PieceSet("Fantasy", "Maurizio Monge", "MIT", "fantasy/", "1"),
        "papercut" to PieceSet("Papercut", "Nikolay Anzarov", "CC BY 4.0", "papercut/", "1"),
        "mpchess" to PieceSet("MPChess", "Maxime Chupin", "GPLv3+", "mpchess/", "1"),
        "pirouetti" to PieceSet("Pirouetti", "pirouetti", "AGPLv3+", "pirouetti/", "1"),
        "rhosgfx" to PieceSet("RhosGFX", "RhosGFX", "CC0 (public domain)", "rhosgfx/", "1"),
        "totoy" to PieceSet("Totoy", "Kosal Sen", "CC BY 4.0", "totoy/", "1"),
        "kiwen-suwi" to PieceSet("Kiwen-suwi", "neverRare", "CC BY 4.0", "kiwen-suwi/", "1"),
        "firi" to PieceSet("Firi", "James Faure", "CC BY 4.0", "firi/", "1"),
        "pixel" to PieceSet("Pixel", "therealqtpi", "AGPLv3+", "pixel/", "1"),
        "shapes" to PieceSet("Shapes", "flugsio", "CC BY-SA 4.0", "shapes/", "1"),
        "letter" to PieceSet("Letters", "usolando", "AGPLv3+", "letter/", "1"),
    )

    private fun hex(rgb: IntArray): String = "#%02x%02x%02x".format(rgb[0], rgb[1], rgb[2])

    private fun rgb(h: String): IntArray =
        intArrayOf(h.substring(1, 3).toInt(16), h.substring(3, 5).toInt(16), h.substring(5, 7).toInt(16))

    private fun odd(h: String): String = rgb(h).let { hex(intArrayOf(it[0], it[1], it[2] + SHADE_STEP)) }

    /**
     * A square lit for the last move: the square half-way to a warm yellow, as a translucent
     * yellow over it would look; the blue kept clear of the signature's headroom. `_highlight`.
     */
    private fun highlight(h: String): String {
        val c = rgb(h)
        val y = intArrayOf(255, 255, 51)
        val mixed = IntArray(3) { Math.rint(c[it] + (y[it] - c[it]) * 0.5).toInt() }
        return hex(intArrayOf(mixed[0], mixed[1], minOf(mixed[2], 255 - SHADE_STEP)))
    }

    class BoardTheme(val label: String, val light: String, val dark: String, val lightHl: String = "", val darkHl: String = "") {
        fun colours(): JSONObject {
            val lightHl = lightHl.ifEmpty { highlight(light) }
            val darkHl = darkHl.ifEmpty { highlight(dark) }
            return JSONObject()
                .put("light", light)
                .put("light_odd", odd(light))
                .put("dark", dark)
                .put("dark_odd", odd(dark))
                .put("light_hl", lightHl)
                .put("light_hl_odd", odd(lightHl))
                .put("dark_hl", darkHl)
                .put("dark_hl_odd", odd(darkHl))
        }
    }

    // Plain colours: nothing here is anyone's artwork. Green is the panel's first board, its
    // highlights as they were.
    val BOARD_THEMES: Map<String, BoardTheme> = linkedMapOf(
        "green" to BoardTheme("Green", "#eeeed2", "#769656", "#f6f669", "#baca2b"),
        "brown" to BoardTheme("Brown", "#f0d9b5", "#b58863"),
        "blue" to BoardTheme("Blue", "#dee3e6", "#8ca2ad"),
        "ocean" to BoardTheme("Ocean", "#eae9d2", "#4b7399"),
        "grey" to BoardTheme("Grey", "#e3e3e3", "#a1a1a1"),
        "purple" to BoardTheme("Purple", "#eeeaf2", "#8877b7"),
        "olive" to BoardTheme("Olive", "#e8e7c8", "#8a9153"),
        "sand" to BoardTheme("Sand", "#f2e4c3", "#c9a26b"),
        "walnut" to BoardTheme("Walnut", "#e3c49a", "#8b5a3c"),
        "cherry" to BoardTheme("Cherry", "#f1d9c9", "#b56060"),
        "coral" to BoardTheme("Coral", "#f4e3d7", "#e07a5f"),
        "ice" to BoardTheme("Ice", "#eef3f7", "#a3bfd6"),
        "tournament" to BoardTheme("Tournament", "#eeeee6", "#3d6b50"),
        "teal" to BoardTheme("Teal", "#e0ece8", "#4f8a83"),
        "rose" to BoardTheme("Rose", "#f3e3e6", "#c07a8a"),
        "night" to BoardTheme("Night", "#9aa3ad", "#4a5460"),
    )

    // ---- a player's own pieces ----

    const val MAX_BYTES = 2_000_000 // an image
    private val NAMES = mapOf('k' to "K", 'q' to "Q", 'r' to "R", 'b' to "B", 'n' to "N", 'p' to "P")
    private val WORDS = linkedMapOf("king" to "K", "queen" to "Q", "rook" to "R", "bishop" to "B", "knight" to "N", "pawn" to "P")
    private val EXT = mapOf("image/png" to ".png", "image/svg+xml" to ".svg", "image/webp" to ".webp", "image/jpeg" to ".jpg")
    private val SHORT = Regex("([wb])[-_ ]?([kqrbnp])")
    private val WIKIMEDIA = Regex("chess_([kqrbnp])([ld])t\\d*")

    /** `Path(name).stem`: the last part of the path without its last suffix. */
    private fun stem(filename: String): String {
        val name = filename.substringAfterLast('/').substringAfterLast('\\')
        val dot = name.lastIndexOf('.')
        return if (dot > 0 && dot < name.length - 1) name.substring(0, dot) else name
    }

    /**
     * Which piece a file is, from its name: "wK.png", "bq.svg", "w_n.png", "white-knight.png",
     * "knight_black.svg", "Chess_nlt45.svg" (the Wikimedia naming: l is white, d is black).
     * `piece_code`.
     */
    fun pieceCode(filename: String): String? {
        val stem = stem(filename).lowercase()
        SHORT.matchEntire(stem)?.let { return it.groupValues[1] + NAMES.getValue(it.groupValues[2][0]) }
        WIKIMEDIA.find(stem)?.let { return (if (it.groupValues[2] == "l") "w" else "b") + NAMES.getValue(it.groupValues[1][0]) }
        val colour = if ("white" in stem) "w" else if ("black" in stem) "b" else null
        val kinds = WORDS.filter { (word, _) -> word in stem }.values.toList()
        if (colour != null && kinds.size == 1) return colour + kinds[0]
        return null
    }

    private fun startsWith(data: ByteArray, prefix: ByteArray, at: Int = 0): Boolean =
        data.size >= at + prefix.size && prefix.indices.all { data[at + it] == prefix[it] }

    /** The image type from its first bytes: PNG, JPEG, WebP or SVG; else null. `sniff`. */
    fun sniff(data: ByteArray): String? {
        if (startsWith(data, byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 13, 10, 26, 10))) {
            return "image/png"
        }
        if (startsWith(data, byteArrayOf(0xff.toByte(), 0xd8.toByte(), 0xff.toByte()))) return "image/jpeg"
        if (startsWith(data, "RIFF".toByteArray()) && startsWith(data, "WEBP".toByteArray(), 8)) return "image/webp"
        val head = String(data, 0, minOf(2048, data.size), Charsets.ISO_8859_1).trimStart(' ', '\t', '\n', '\r', '\u000b', '\u000c').lowercase()
        val start = String(data, 0, minOf(4096, data.size), Charsets.ISO_8859_1).lowercase()
        if ((head.startsWith("<svg") || head.startsWith("<?xml") || head.startsWith("<!--")) && "<svg" in start) {
            return "image/svg+xml"
        }
        return null
    }

    const val LEGACY_NAME = "My pieces"
    const val DEFAULT_NAME = "My set"
    const val REMOVED = ".removed"
    const val UNDO_S = 600 // a deleted set can come back this long
    const val MAX_NAME = 40
    private val ID = Regex("[a-z0-9][a-z0-9-]{0,39}")

    fun ownRoot(files: File): File = File(File(files, "pieces"), "own")

    /**
     * The own set a board setting names ("own:<id>", or "custom", the one own set before there
     * were several), or null for a shipped set. `own_id`.
     */
    fun ownId(pieces: String): String? {
        if (pieces == CUSTOM) return CUSTOM
        if (pieces.startsWith(OWN) && ID.matches(pieces.removePrefix(OWN))) return pieces.removePrefix(OWN)
        return null
    }

    private fun meta(folder: File): JSONObject =
        runCatching { JSONObject(File(folder, "set.json").readText()) }.getOrElse { JSONObject() }

    private fun writeMeta(folder: File, name: String, order: Int) {
        File(folder, "set.json").writeText(JSONObject().put("name", name).put("order", order).toString())
    }

    /** `" ".join(str(name).split())[:MAX_NAME].strip()`. */
    private fun cleanName(name: Any?): String {
        val text = if (name == null || name == JSONObject.NULL) "" else name.toString()
        return text.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ").take(MAX_NAME).trim()
    }

    private fun slug(name: String): String =
        name.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-').take(32).trim('-').ifEmpty { "set" }

    private fun images(folder: File): Map<String, String> {
        val out = LinkedHashMap<String, String>()
        if (!folder.isDirectory) return out
        for (path in folder.listFiles().orEmpty().sortedBy { it.name }) {
            val code = stem(path.name)
            if (code !in CODES || !path.isFile) continue
            val data = path.readBytes()
            val kind = sniff(data) ?: continue
            out[code] = "data:$kind;base64," + Base64.getEncoder().encodeToString(data)
        }
        return out
    }

    /**
     * The one own set there was before (pieces/custom) as the set "My pieces": copied, the copy
     * read back, then the old files removed; nothing is lost if a copy is not the same.
     * `_migrate`.
     */
    private fun migrate(files: File) {
        val old = File(File(files, "pieces"), CUSTOM)
        if (!old.isDirectory) return
        val pieces = old.listFiles().orEmpty().filter { it.isFile && stem(it.name) in CODES }
        val new = File(ownRoot(files), CUSTOM)
        if (pieces.isEmpty() || new.exists()) return
        new.mkdirs()
        for (p in pieces) p.copyTo(File(new, p.name))
        if (pieces.all { File(new, it.name).readBytes().contentEquals(it.readBytes()) }) {
            writeMeta(new, LEGACY_NAME, 0)
            for (p in pieces) p.delete()
            if (old.listFiles().isNullOrEmpty()) old.delete() // nothing else was in it
        } else {
            new.deleteRecursively()
        }
    }

    /** Sets deleted longer ago than their Undo, gone for good. `_purge`. */
    private fun purge(files: File) {
        val bin = File(ownRoot(files), REMOVED)
        if (!bin.isDirectory) return
        val now = System.currentTimeMillis()
        for (d in bin.listFiles().orEmpty()) {
            if (d.isDirectory && now - d.lastModified() > UNDO_S * 1000L) d.deleteRecursively()
        }
    }

    class OwnSet(val id: String, val name: String, val order: Int)

    /** The player's own sets, in the order they were added. `own_sets`. */
    fun ownSets(files: File): List<OwnSet> {
        migrate(files)
        purge(files)
        val root = ownRoot(files)
        if (!root.isDirectory) return emptyList()
        val out = ArrayList<OwnSet>()
        for (d in root.listFiles().orEmpty()) {
            if (d.isDirectory && ID.matches(d.name)) {
                val m = meta(d)
                val name = cleanName(m.opt("name")).ifEmpty { d.name }
                out.add(OwnSet(d.name, name, m.optInt("order", 0)))
            }
        }
        return out.sortedWith(compareBy<OwnSet>({ it.order }, { it.name.lowercase() }, { it.id }))
    }

    /** An own set's pieces as data URLs, by code: the ones it has. `own_images`. */
    fun ownImages(files: File, setId: String): Map<String, String> {
        migrate(files)
        if (!ID.matches(setId)) return emptyMap()
        return images(File(ownRoot(files), setId))
    }

    /**
     * Each image a piece names kept in the set's folder (in place of that piece's last one);
     * what was not taken, and why. `_save`.
     */
    private fun save(folder: File, given: JSONArray): Pair<List<String>, JSONArray> {
        val taken = ArrayList<String>()
        val skipped = JSONArray()
        for (i in 0 until given.length()) {
            val f = given.optJSONObject(i) ?: JSONObject()
            val name = f.optString("name", "")
            fun skip(why: String) = skipped.put(JSONObject().put("name", name).put("why", why))
            val code = pieceCode(name)
            if (code == null) {
                skip("the name does not say which piece it is")
                continue
            }
            val data = try {
                Base64.getDecoder().decode(f.optString("data", ""))
            } catch (_: IllegalArgumentException) {
                skip("it could not be read")
                continue
            }
            if (data.size > MAX_BYTES) {
                skip("it is larger than 2 MB")
                continue
            }
            val kind = sniff(data)
            if (kind == null) {
                skip("it is not a PNG, JPEG, WebP or SVG image")
                continue
            }
            folder.mkdirs()
            for (old in folder.listFiles().orEmpty()) if (stem(old.name) == code) old.delete()
            File(folder, code + EXT.getValue(kind)).writeBytes(data)
            taken.add(code)
        }
        return taken to skipped
    }

    private fun result(files: File, setId: String, taken: List<String>, skipped: JSONArray): JSONObject {
        val have = ownImages(files, setId).keys
        return JSONObject()
            .put("ok", taken.isNotEmpty())
            .put("id", OWN + setId)
            .put("name", cleanName(meta(File(ownRoot(files), setId)).opt("name")).ifEmpty { setId })
            .put("taken", JSONArray(taken))
            .put("skipped", skipped)
            .put("missing", JSONArray(CODES.filter { it !in have }))
    }

    private fun missing(setId: String): JSONObject =
        JSONObject().put("ok", false).put("id", OWN + setId).put("error", "There is no such set.")

    /**
     * A new own set from the images chosen, named (made unique: "My set 2"); nothing is made
     * when no image was taken. `create_own`.
     */
    fun createOwn(files: File, name: String, given: JSONArray): JSONObject {
        val sets = ownSets(files)
        val names = sets.map { it.name.lowercase() }.toSet()
        val base = cleanName(name).ifEmpty { DEFAULT_NAME }
        var shown = base
        var n = 2
        while (shown.lowercase() in names) {
            shown = "${base.take(MAX_NAME - 4)} $n"
            n += 1
        }
        val root = ownRoot(files)
        var setId = slug(shown)
        var k = 2
        while (File(root, setId).exists() || File(File(root, REMOVED), setId).exists()) {
            setId = "${slug(shown)}-$k"
            k += 1
        }
        val folder = File(root, setId)
        val (taken, skipped) = save(folder, given)
        if (taken.isEmpty()) {
            folder.deleteRecursively()
            return JSONObject().put("ok", false).put("taken", JSONArray()).put("skipped", skipped)
                .put("missing", JSONArray(CODES))
        }
        writeMeta(folder, shown, (sets.maxOfOrNull { it.order } ?: 0) + 1)
        return result(files, setId, taken, skipped)
    }

    /** Images added to an own set, each in place of its piece's last one. `add_images`. */
    fun addImages(files: File, setId: String, given: JSONArray): JSONObject {
        val folder = File(ownRoot(files), setId)
        if (!ID.matches(setId) || !folder.isDirectory) return missing(setId)
        val (taken, skipped) = save(folder, given)
        return result(files, setId, taken, skipped)
    }

    /** `rename_own`. */
    fun renameOwn(files: File, setId: String, name: String): JSONObject {
        val folder = File(ownRoot(files), setId)
        if (!ID.matches(setId) || !folder.isDirectory) return missing(setId)
        val shown = cleanName(name)
        if (shown.isEmpty()) return JSONObject().put("ok", false).put("id", OWN + setId).put("error", "A set needs a name.")
        writeMeta(folder, shown, meta(folder).optInt("order", 0))
        return JSONObject().put("ok", true).put("id", OWN + setId).put("name", shown)
    }

    /** An own set put aside for its Undo ([UNDO_S]), then gone. `delete_own`. */
    fun deleteOwn(files: File, setId: String): JSONObject {
        val root = ownRoot(files)
        val folder = File(root, setId)
        if (!ID.matches(setId) || !folder.isDirectory) return missing(setId)
        val aside = File(File(root, REMOVED), setId)
        aside.parentFile?.mkdirs()
        if (aside.exists()) aside.deleteRecursively()
        if (!folder.renameTo(aside)) return JSONObject().put("ok", false).put("id", OWN + setId).put("error", "Could not delete it.")
        aside.setLastModified(System.currentTimeMillis())
        return JSONObject().put("ok", true).put("id", OWN + setId)
    }

    /** A set deleted within its Undo, back as it was. `restore_own`. */
    fun restoreOwn(files: File, setId: String): JSONObject {
        val root = ownRoot(files)
        val aside = File(File(root, REMOVED), setId)
        if (!ID.matches(setId) || !aside.isDirectory || File(root, setId).exists() || !aside.renameTo(File(root, setId))) {
            return JSONObject().put("ok", false).put("id", OWN + setId).put("error", "That set was deleted too long ago to put back.")
        }
        return JSONObject().put("ok", true).put("id", OWN + setId)
    }

    // ---- what the pages read ----

    /**
     * The boards' look now: an image for every piece (a player's own, where they gave one, else
     * Sidecoach's), and the theme's colours. `look`.
     */
    fun look(pieces: String, theme: String, files: File): JSONObject {
        val base = PIECE_SETS.getValue(DEFAULT_SET)
        val images = LinkedHashMap<String, String>()
        for (c in CODES) images[c] = base.url(c)
        val set = PIECE_SETS[pieces]
        val own = ownId(pieces)
        if (own != null) images.putAll(ownImages(files, own))
        else if (set != null) for (c in CODES) images[c] = set.url(c)
        val board = BOARD_THEMES[theme] ?: BOARD_THEMES.getValue(DEFAULT_THEME)
        val colours = board.colours()
        return JSONObject()
            .put("pieces", pieces)
            .put("theme", theme)
            .put("images", JSONObject(images as Map<*, *>))
            .put("board", colours)
            .put("stamp", stamp(pieces, theme, images, colours))
    }

    /**
     * A short hash of the look: it changes with the set, the colours or any image of an own set.
     * Every page reads it every few seconds and reads the look again when it changes, so it
     * follows the settings whatever changed them. `stamp`.
     */
    fun stamp(pieces: String, theme: String, images: Map<String, String>, colours: JSONObject): String {
        val parts = ArrayList<String>()
        parts.add(pieces)
        parts.add(theme)
        for (c in CODES) parts.add("$c=${images[c] ?: ""}")
        for (k in colours.keys().asSequence().sorted()) parts.add("$k=${colours.getString(k)}")
        val digest = java.security.MessageDigest.getInstance("SHA-1").digest(parts.joinToString("\n").toByteArray(Charsets.UTF_8))
        return digest.joinToString("") { "%02x".format(it) }.take(16)
    }

    /**
     * What the settings page offers: every set with its preview images, author and licence,
     * every theme with its colours, and the player's own sets, each with its pieces and what it
     * lacks (Sidecoach's stand in). `catalogue`.
     */
    fun catalogue(files: File): JSONObject {
        val own = JSONArray()
        for (o in ownSets(files)) {
            val images = ownImages(files, o.id)
            own.put(
                JSONObject().put("name", OWN + o.id).put("label", o.name)
                    .put("images", JSONObject(images as Map<*, *>))
                    .put("missing", JSONArray(CODES.filter { it !in images })),
            )
        }
        return JSONObject()
            .put(
                "sets",
                JSONArray(
                    PIECE_SETS.map { (n, s) ->
                        JSONObject().put("name", n).put("label", s.label).put("author", s.author).put("licence", s.licence)
                            .put("images", JSONObject().apply { for (c in CODES) put(c, s.url(c)) })
                    },
                ),
            )
            .put(
                "themes",
                JSONArray(
                    BOARD_THEMES.map { (n, t) ->
                        JSONObject().put("name", n).put("label", t.label).apply {
                            val colours = t.colours()
                            for (k in colours.keys()) put(k, colours.get(k))
                        }
                    },
                ),
            )
            .put("own", own)
    }
}
