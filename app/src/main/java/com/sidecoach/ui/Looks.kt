package com.sidecoach.ui

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Base64

/**
 * How the boards look (`sidecoach/looks.py`): the piece sets Sidecoach ships, the board colour
 * themes, and a player's own piece images.
 *
 * The shipped sets are Sidecoach's own (the default) and every Lichess set whose licence allows
 * any use: static files beside the page, under /pieces/ and /pieces/<set>/. The themes are
 * plain colours. A player's own pieces are images they choose, kept only in the app's files
 * (pieces/custom) and handed to the page inline. Nothing names or fetches any other product's
 * set.
 *
 * The board's squares carry the panel's signature for the desktop's vision: its odd rows three
 * units bluer than its even rows, square colour and highlight alike; every theme keeps it.
 */
object Looks {
    const val SHADE_STEP = 3
    val CODES: List<String> = "wb".flatMap { c -> "KQRBNP".map { p -> "$c$p" } }
    const val DEFAULT_SET = "sidecoach"
    const val CUSTOM = "custom"
    const val DEFAULT_THEME = "green"

    class PieceSet(val label: String, val author: String, val licence: String, val folder: String, val version: String) {
        fun url(code: String): String = "/pieces/$folder$code.svg?v=$version"
    }

    val PIECE_SETS: Map<String, PieceSet> = linkedMapOf(
        "sidecoach" to PieceSet("Sidecoach", "Sidecoach", "Sidecoach's own", "", "own1"),
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

    fun customDir(files: File): File = File(File(files, "pieces"), CUSTOM)

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

    /** The player's own pieces as data URLs, by code: the ones there are. `custom_images`. */
    fun customImages(files: File): Map<String, String> {
        val folder = customDir(files)
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
     * Images the player chose, each {"name", "data" (base64)}: each one a piece names is kept
     * (in place of that piece's last one); what was not taken is said, and why. `save_custom`.
     */
    fun saveCustom(files: File, given: JSONArray): JSONObject {
        val folder = customDir(files)
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
        val have = customImages(files).keys
        return JSONObject()
            .put("ok", taken.isNotEmpty())
            .put("taken", JSONArray(taken))
            .put("skipped", skipped)
            .put("missing", JSONArray(CODES.filter { it !in have }))
    }

    /** The player's own pieces gone (the files one by one; the folder stays). `clear_custom`. */
    fun clearCustom(files: File) {
        val folder = customDir(files)
        if (folder.isDirectory) for (path in folder.listFiles().orEmpty()) if (path.isFile) path.delete()
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
        if (pieces == CUSTOM) images.putAll(customImages(files))
        else if (set != null) for (c in CODES) images[c] = set.url(c)
        val board = BOARD_THEMES[theme] ?: BOARD_THEMES.getValue(DEFAULT_THEME)
        return JSONObject()
            .put("pieces", pieces)
            .put("theme", theme)
            .put("images", JSONObject(images as Map<*, *>))
            .put("board", board.colours())
    }

    /**
     * What the settings page offers: every set with its preview images, author and licence,
     * every theme with its colours, and the player's own pieces as they are. `catalogue`.
     */
    fun catalogue(files: File): JSONObject {
        val own = customImages(files)
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
            .put(
                "custom",
                JSONObject().put("images", JSONObject(own as Map<*, *>)).put("missing", JSONArray(CODES.filter { it !in own })),
            )
    }
}
