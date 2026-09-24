package com.sidecoach.coach

import com.sidecoach.analysis.sanLine
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank

/**
 * The idea of the best line ([Idea]) in the coach's words, one field per rung. A port of
 * `sidecoach/coach/idea_words.py`.
 *
 * Every sentence here is built from the board and the engine's line, and says something true
 * about this position that helps the player find the answer without handing it over: the goal
 * and the pattern with why it works (the idea rung), then the target and the piece (the focus
 * rung), then the move played out (the move rung). A sentence that could be said of any
 * position is not written here (Tai, 2026-09-23: "we want instructive comments, not generic").
 * These are spoken to the player, so every table is the desktop's, word for word.
 */
object IdeaWords {

    val PIECE_NAMES = mapOf(
        PAWN to "pawn",
        KNIGHT to "knight",
        BISHOP to "bishop",
        ROOK to "rook",
        QUEEN to "queen",
        KING to "king",
    )
    private val ORDINAL = mapOf(1 to "first", 2 to "second", 3 to "third", 4 to "fourth", 5 to "fifth")
    private val NUMBER = mapOf(
        1 to "one", 2 to "two", 3 to "three", 4 to "four", 5 to "five", 6 to "six", 7 to "seven",
    )

    /** The pattern's name, as the coach names it. */
    val PATTERN_NAMES: Map<String, String> = linkedMapOf(
        "smotheredMate" to "a smothered mate",
        "backRankMate" to "a back-rank mate",
        "anastasiaMate" to "an Anastasia mate",
        "arabianMate" to "an Arabian mate",
        "hookMate" to "a hook mate",
        "bodenMate" to "Boden's mate",
        "doubleBishopMate" to "a mate by two bishops",
        "dovetailMate" to "a dovetail mate",
        "vukovicMate" to "Vukovic's mate",
        "killBoxMate" to "a box mate",
        "swallowstailMate" to "a swallow's tail mate",
        "epauletteMate" to "an epaulette mate",
        "triangleMate" to "a triangle mate",
        "blindSwineMate" to "a blind swine mate",
        "cornerMate" to "a corner mate",
        "operaMate" to "an opera mate",
        "pillsburysMate" to "Pillsbury's mate",
        "morphysMate" to "Morphy's mate",
        "balestraMate" to "a Balestra mate",
        "mate" to "a mating attack",
        "doubleCheck" to "a double check",
        "fork" to "a fork",
        "skewer" to "a skewer",
        "discoveredAttack" to "a discovered attack",
        "absolutePin" to "a pin",
        "pin" to "a pin",
        "trappedPiece" to "a trapped piece",
        "capturingDefender" to "removing the defender",
        "deflection" to "a deflection",
        "attraction" to "a decoy",
        "hangingPiece" to "a loose piece",
    )

    /** Why each pattern works, in general terms, naming none of the player's own pieces. */
    val PATTERN_WHY: Map<String, String> = linkedMapOf(
        "smotheredMate" to "their king is hemmed in by its own pieces, so a check it can neither " +
            "block nor step away from ends it",
        "backRankMate" to "their king is shut in behind its own pawns, and a check along the back " +
            "rank has nowhere to be escaped",
        "anastasiaMate" to "one piece takes the king's escape squares while another mates it along " +
            "the edge of the board",
        "arabianMate" to "with the king in a corner, one piece guards its way out and another mates " +
            "right beside it",
        "hookMate" to "a protected piece next to the king cuts off its escape while the mate comes " +
            "from close by",
        "bodenMate" to "two long diagonals cross at the king, and its own pieces block its way out",
        "doubleBishopMate" to "two diagonals hit the king and the squares it would run to",
        "dovetailMate" to "the mating piece stands right next to the king, its retreat blocked by " +
            "its own pieces",
        "vukovicMate" to "one piece mates with the support of another beside the king on the edge",
        "killBoxMate" to "the king is caught in a box of squares no escape leads out of",
        "swallowstailMate" to "the check comes from right in front of the king, guarded, and the two " +
            "squares behind it are blocked by its own pieces",
        "epauletteMate" to "the king's own pieces stand on both sides of it like epaulettes, so a " +
            "check along the line leaves it nowhere to go",
        "triangleMate" to "two pieces stand next to the king in a triangle, each guarding the " +
            "other and every square around it",
        "blindSwineMate" to "two pieces side by side deep in their camp sweep the king's squares, " +
            "each protecting the other",
        "cornerMate" to "the king is shut in the corner, and a check it cannot step out of or " +
            "block finishes it",
        "operaMate" to "the check comes from right beside the king, guarded from a distance, " +
            "and the king's own piece takes away its escape",
        "pillsburysMate" to "the check comes down a file while the square the king would run to is " +
            "covered from a diagonal",
        "morphysMate" to "the check comes down a long diagonal while a file shuts off the king's escape",
        "balestraMate" to "the check comes from a distance and every square the king could reach " +
            "is already covered",
        "doubleCheck" to "two pieces give check at once, so the king must move and they cannot " +
            "answer anything else",
        "fork" to "one piece attacks two of theirs at once, and they can save only one",
        "skewer" to "a valuable piece is attacked on a line, and when it steps aside the piece " +
            "behind it falls",
        "discoveredAttack" to "one piece steps off a line with a threat of its own and uncovers an " +
            "attack from the piece behind it: two threats at once",
        "absolutePin" to "one of their pieces cannot move without exposing the king behind it, so " +
            "it can be attacked again and again",
        "pin" to "one of their pieces cannot move without losing what stands behind it, so it can " +
            "be attacked again",
        "trappedPiece" to "one of their pieces has no safe square left, so an attack on it wins it",
        "capturingDefender" to "the piece that guards the target is taken first, and the target falls",
        "deflection" to "a defender is pulled away from what it guards, and what it guarded falls",
        "attraction" to "a piece is lured onto a square where it can be hit",
        "hangingPiece" to "something of theirs is not defended at all",
    )

    /** What each preparation does, said of the moves before the blow. */
    val PREPARATION_WHY: Map<String, String> = linkedMapOf(
        "sacrifice" to "it starts by giving up material",
        "attraction" to "their king or a defender is lured onto the wrong square first",
        "deflection" to "a defender is pulled away first",
        "capturingDefender" to "a defender is taken first",
        "clearance" to "a line or a square is cleared first",
    )

    /** What a preparation has done, said once the blow comes. */
    val PREPARED: Map<String, String> = linkedMapOf(
        "sacrifice" to "you have given up material to open the way",
        "attraction" to "a piece has been lured onto the wrong square",
        "deflection" to "a defender has been pulled away",
        "capturingDefender" to "a defender has been taken",
        "clearance" to "a line has been cleared",
    )

    /** The same, when it is the first move itself. */
    val STARTS: Map<String, String> = linkedMapOf(
        PREPARED.getValue("sacrifice") to "by giving up material to open the way",
        PREPARED.getValue("attraction") to "by luring a piece onto the wrong square",
        PREPARED.getValue("deflection") to "by pulling a defender away",
        PREPARED.getValue("capturingDefender") to "by taking a defender",
        PREPARED.getValue("clearance") to "by clearing a line",
        "a trade has cleared the way" to "with a trade",
    )

    /** The question that sets the player looking, per pattern. */
    val PATTERN_QUESTION: Map<String, String> = linkedMapOf(
        "fork" to "Which two of their pieces could one of yours hit at once, and from where?",
        "skewer" to "Which line has two of their pieces on it, the more valuable in front?",
        "discoveredAttack" to "Which of your pieces is standing in front of another's line?",
        "doubleCheck" to "Which move uncovers a check while giving one itself?",
        "absolutePin" to "Which of their pieces could you pin to their king -- and what can it " +
            "no longer do then?",
        "pin" to "Which of their pieces stands in front of a bigger one, on a line one of yours " +
            "could take?",
        "trappedPiece" to "Which of their pieces is short of squares?",
        "capturingDefender" to "Which of their pieces is doing the guarding?",
        "deflection" to "Which of their pieces is guarding two things at once?",
        "attraction" to "Onto which square would you like to drag one of their pieces?",
        "hangingPiece" to "Which of their pieces has no defender?",
    )

    /** The plan's why, keyed by the positional idea's kind (`analysis/positional.py`). */
    val PLAN_WHY: Map<String, String> = linkedMapOf(
        "castle" to "a king left in the centre gets caught when the position opens, and castling " +
            "also brings a rook into play",
        "rescue" to "a piece left attacked is simply lost, so it moves first -- ideally to a square " +
            "where it does something",
        "prophylaxis" to "their idea is worth stopping before it starts",
        "activateKing" to "in an endgame the king is a fighting piece, not one to hide",
        "develop" to "pieces still at home take no part in the game",
        "outpostMove" to "a piece on a square no enemy pawn can attack stays there for good",
        "centralise" to "a piece in the centre reaches both sides of the board",
        "takeFile" to "a rook on an open file gets into their position",
        "rookBehindPasser" to "a rook behind a passed pawn pushes it and grows stronger as it goes",
        "break" to "a pawn break opens lines for your pieces",
        "centre" to "the centre is where pieces have the most room",
        "pawnStorm" to "pawns rolling at their king open lines toward it",
        "pushPasser" to "a passed pawn is there to be pushed",
        "passerPawnEnding" to "in a pawn ending the first passed pawn usually decides, and the " +
            "king walks it home",
        "activate" to "your least active piece is the one worth improving",
        "space" to "more space gives your pieces room and cramps theirs",
        "simplify" to "when you are ahead, trades make the extra material count",
    )
    val PLAN_QUESTION: Map<String, String> = linkedMapOf(
        "castle" to "Is there any reason to keep your king in the centre?",
        "rescue" to "Which of your pieces is attacked, and where is it safe and useful?",
        "prophylaxis" to "If you passed, what would your opponent play?",
        "activateKing" to "Where would your king do the most good?",
        "develop" to "Which of your pieces is still at home, and where does it belong?",
        "outpostMove" to "Which square in their half can no pawn of theirs ever attack?",
        "centralise" to "Which of your pieces could reach the centre?",
        "takeFile" to "Which file has no pawns on it?",
        "rookBehindPasser" to "Where does your rook best support the passed pawn?",
        "break" to "Which pawn move would open lines toward their weaknesses?",
        "centre" to "What would make the centre yours?",
        "pawnStorm" to "Which pawn can start rolling toward their king?",
        "pushPasser" to "What stops your passed pawn from advancing?",
        "activate" to "Which of your pieces has no job right now?",
        "space" to "Where can you take space without leaving a weakness behind?",
        "simplify" to "Which trade makes your extra material count?",
        "passerPawnEnding" to "Which pawn move makes you a passed pawn -- and can their king catch it?",
    )

    val NUDGE: Map<String, String> = linkedMapOf(
        "fork" to "Which square hits both?",
        "skewer" to "Which check or attack lines them up?",
        "discoveredAttack" to "Which move uncovers the line with a threat of its own?",
        "doubleCheck" to "Which move gives check twice?",
        "absolutePin" to "Which move pins it?",
        "pin" to "Which move pins it?",
        "trappedPiece" to "How do you attack it?",
        "hangingPiece" to "How do you win it safely?",
        "capturingDefender" to "Which capture removes the guard?",
        "deflection" to "How do you pull it away?",
        "attraction" to "What drags it there?",
    )

    /** `facts.describe_tactic`, handed in as the desktop hands it to `why`. */
    fun interface Describe {
        fun describe(board: Board, name: String, squares: List<String>, bestMove: Move?): String?
    }

    private fun sq(square: Int?): String = if (square != null) squareName(square) else ""

    private fun named(board: Board, square: Int): String {
        val piece = board.pieceAt(square) ?: return ""
        return "${PIECE_NAMES.getValue(piece.pieceType)} on ${squareName(square)}"
    }

    /** The pattern's name: "a fork", "a smothered mate", "a mating attack". */
    fun pattern(idea: Idea): String? {
        if (idea.blow != null) return PATTERN_NAMES[idea.blow.name]
        if (idea.goal == "mate") return PATTERN_NAMES["mate"]
        return null
    }

    /** What the best line is for, naming no square, piece or move. */
    fun goal(idea: Idea): String? {
        if (idea.goal == "mate") {
            val n = idea.mateIn ?: 0
            if (n == 1) return "you can mate in one move"
            return if (n <= 5) "you can force mate in ${NUMBER[n] ?: n.toString()}" else "you can force mate"
        }
        if (idea.goal == "draw") {
            return when (idea.draw ?: "") {
                "perpetual" -> "you can save the game: check after check, and they cannot escape"
                "stalemate" -> "you can save the game with a stalemate"
                else -> "you can hold the game: the same position comes round again"
            }
        }
        if (idea.goal == "win" && idea.level) {
            // Winning it back, not winning: the engine calls the game level after it.
            val what = idea.won ?: "material"
            if (idea.down) return "you can win back $what, and that levels the game"
            return "you can win $what, and that holds the game level"
        }
        if (idea.goal == "win") {
            return if (!idea.won.isNullOrEmpty()) "you can win ${idea.won}" else "you can win material by force"
        }
        if (idea.goal == "promote") return "you can push a pawn through to promotion"
        if (idea.goal == "attack") {
            return if (idea.winning) {
                "you have a winning attack on their king"
            } else {
                "you have a strong attack on their king"
            }
        }
        if (idea.goal == "convert") {
            return "you can trade into a won ${if (!idea.ending.isNullOrEmpty()) idea.ending else "simpler"} ending"
        }
        if (idea.goal == "king") return "you are winning, and your king leads the way"
        if (idea.goal == "breakthrough") return "you can break through and make a pawn that runs"
        if (idea.goal == "combination") return "you can give up material and win it back with a tactic"
        if (idea.goal == "defend") {
            val lead = if (idea.winning) "you are winning, but " else ""
            if (idea.threatened == "mate") return "${lead}your opponent threatens mate: meet it first"
            return "${lead}your opponent threatens to win material: meet it first"
        }
        if (idea.goal == "plan" && idea.winning) {
            // Won, with nothing to take and no mate: said so, then how -- never only "a quiet
            // move is wanted" in a position the engine calls won (Lichess 8KhlA, y6QH7).
            return "you are winning: no trick is needed here, only good technique"
        }
        return null
    }

    /**
     * The pattern and why it works here, square-free: the idea rung's teaching -- and what else
     * makes it work: a guard that is pinned, a quiet first move, an in-between move.
     */
    fun mechanism(idea: Idea): String? {
        val text = mechanismText(idea)
        if (text == null || idea.goal !in listOf("mate", "win", "promote", "combination", "attack")) return text
        val extra = ArrayList<String>()
        if (idea.also != null && idea.also in PATTERN_NAMES) {
            extra.add("There is ${PATTERN_NAMES.getValue(idea.also)} in it too")
        }
        if (idea.pinnedGuard != null) extra.add("It works because the piece that could take is pinned")
        if (idea.startsWith == "quiet" && idea.sacrifices.isEmpty()) {
            extra.add(
                "It starts with a quiet move: no check and no capture, but a threat they cannot meet",
            )
        }
        if (idea.intermezzo) extra.add("On the way comes an in-between move: a check before taking back")
        return if (extra.isNotEmpty()) (listOf(text) + extra).joinToString(". ") else text
    }

    /** `idea_words._mechanism`: the pattern and why it works here, square-free. */
    private fun mechanismText(idea: Idea): String? {
        val name = idea.blow?.name
        val parts = ArrayList<String>()
        if (name != null && name in PATTERN_WHY) {
            parts.add("${PATTERN_NAMES.getValue(name)}: ${PATTERN_WHY.getValue(name)}")
        } else if (idea.goal == "mate") {
            parts.add("a mating attack: their king runs out of squares")
        } else if (idea.goal == "promote") {
            parts.add("a pawn on the run: once its way is clear it is too far advanced to be caught")
        } else if (idea.goal == "attack") {
            parts.add(
                "an attack on the king: check after check drives it into the open, where your " +
                    "pieces can pick things off",
            )
        } else if (idea.goal == "convert") {
            parts.add(
                "trading down: when you are winning, every trade takes away their chances, and " +
                    if (idea.ending == "pawn") {
                        "a pawn ending is counted, not played -- your king and pawns get there first"
                    } else {
                        "the fewer pieces, the fewer tricks"
                    },
            )
        } else if (idea.goal == "king" && idea.opposition) {
            parts.add(
                "the opposition: with the kings face to face and one square between them, the " +
                    "side that must move gives way",
            )
        } else if (idea.goal == "king") {
            parts.add("king activity: in an endgame the king is a fighting piece, not one to hide")
        } else if (idea.goal == "breakthrough") {
            parts.add("a breakthrough: a pawn given up opens the way, and another pawn runs through")
        } else if (idea.goal == "draw" && idea.draw == "perpetual") {
            parts.add(
                "a perpetual check: their king has nowhere to hide from the checks, and a " +
                    "position that repeats three times is a draw",
            )
        } else if (idea.goal == "draw" && idea.draw == "stalemate") {
            parts.add("a stalemate: if they have no legal move and are not in check, it is a draw")
        } else if (idea.sacrifices.isNotEmpty() && (idea.goal == "win" || idea.goal == "combination")) {
            // No pattern to name, but the sacrifice is the idea: said as the idea.
            val count = idea.sacrifices.size
            val what = if (count == 1) "a sacrifice" else "${NUMBER[count] ?: count.toString()} sacrifices"
            val opened = "you give up material, and what it opens up wins more back"
            if (idea.sacrifices[0].first == 0) return "$what: $opened. It starts straight away"
            val ordinal = ORDINAL[idea.sacrifices[0].first / 2 + 1] ?: "next"
            return "$what: $opened. The first comes on your $ordinal move"
        }
        if (parts.isEmpty()) return null
        var text = parts[0]
        // When it comes, and what it costs: the sacrifices on the way say more than any other
        // preparation (Tai, 2026-09-23: "the coach is completely blind to" them).
        val given = idea.sacrifices
        if (given.isNotEmpty()) {
            val count = given.size
            val ordinal = ORDINAL[given[0].first / 2 + 1] ?: "next"
            val sacs = if (count == 1) "a sacrifice" else "${NUMBER[count] ?: count.toString()} sacrifices"
            if (idea.blowAt > 1 && name != null && name in PATTERN_WHY) {
                val whenText = ORDINAL[idea.blowAt] ?: "next"
                return "$text. It comes on your $whenText move, after $sacs"
            }
            if (given[0].first == 0) {
                if (count == 1) return "$text. It starts with a sacrifice: you give up material to open the way"
                return "$text. It takes $sacs, the first straight away"
            }
            return "$text. It takes $sacs, the first on your $ordinal move"
        }
        // When it comes, and what prepares it: a second sentence. A sacrifice only as the line
        // counts one (`IdeaReader.sacrifices`), never on the tagger's looser word.
        var before: String? = null
        for (prep in idea.preparation) {
            if (prep == "sacrifice") continue
            if (prep in PREPARED && prep != name) {
                before = PREPARED.getValue(prep)
                break
            }
        }
        if (before == null) {
            if (idea.startsWith == "sacrifice") {
                before = PREPARED.getValue("sacrifice")
            } else if (idea.startsWith == "capture" && idea.blowAt > 1) {
                before = "a trade has cleared the way"
            }
        }
        if (idea.goal in listOf("mate", "win", "promote", "attack") &&
            idea.blowAt > 1 &&
            name != null && name in PATTERN_WHY
        ) {
            val whenText = "It comes on your ${ORDINAL[idea.blowAt] ?: "next"} move"
            text += if (!before.isNullOrEmpty()) ". $whenText, once $before" else ". $whenText"
        } else if (before != null && before in STARTS) {
            text += ". It starts ${STARTS.getValue(before)}"
        } else if ((idea.goal == "mate" || idea.goal == "attack") &&
            idea.startsWith == "check" &&
            (idea.goal == "attack" || (idea.mateIn ?: 0) > 1)
        ) {
            text += ". It starts with a check, so they have no time for anything else"
        }
        return text
    }

    /** The idea rung's question: the one that sets the player looking the right way. */
    fun question(idea: Idea, planKind: String?): String {
        val name = idea.blow?.name
        if (idea.goal == "convert") return "Which trade leaves them nothing to fight with?"
        if (idea.goal == "king") {
            if (idea.opposition) return "Where does your king face theirs, so that they must give way?"
            return "Where does your king want to go, and what can it attack there?"
        }
        if (idea.goal == "breakthrough") return "Which pawn can you give up so that another gets through?"
        if (idea.goal == "draw") {
            if (idea.draw == "perpetual") return "Which checks can you keep giving, however their king runs?"
            if (idea.draw == "stalemate") return "What if your king had no move at all -- and you were not in check?"
            return "Which move keeps the position going round in a circle?"
        }
        if (idea.goal == "mate") {
            if (idea.mateIn == 1) return "Which check leaves their king no square at all?"
            if (idea.startsWith == "sacrifice" || idea.sacrifices.isNotEmpty()) {
                return "Go through your checks, the ones that give up material too: after which " +
                    "does their king run out of squares?"
            }
            return "Go through your checks one by one: after each, where can their king go?"
        }
        if ((idea.goal == "win" || idea.goal == "promote") && name != null && name in PATTERN_QUESTION) {
            return PATTERN_QUESTION.getValue(name)
        }
        if (idea.goal == "combination" && name != null && name in PATTERN_QUESTION) {
            return "What could you give up to set it up? " + PATTERN_QUESTION.getValue(name)
        }
        if (idea.goal == "attack") {
            if (idea.startsWith == "sacrifice" || idea.sacrifices.isNotEmpty()) {
                return "Which sacrifice tears open the cover around their king?"
            }
            return "Which check drives their king into the open?"
        }
        if (idea.goal == "win") return "Which of their pieces is loose or attacked more times than it is defended?"
        if (idea.goal == "promote") return "What stands in your pawn's way, and how do you remove it?"
        if (idea.goal == "defend") {
            return "If you passed, what would they play -- and how do you stop it without losing ground?"
        }
        if (!planKind.isNullOrEmpty() && planKind in PLAN_QUESTION) return PLAN_QUESTION.getValue(planKind)
        if (idea.startsWith in listOf("check", "capture", "promotion")) {
            // A forcing answer with no plan behind it: which forcing move, not which idle piece.
            return "Go through your checks and captures: which leaves them the fewest good replies?"
        }
        return "Which of your pieces has no job right now?"
    }

    /** The focus rung's question: the one that leads from the target to the move. */
    fun nudge(idea: Idea): String? {
        val name = idea.blow?.name
        if (idea.goal == "mate") return if (idea.mateIn == 1) "Which check is it?" else "Which check comes first?"
        if (idea.goal == "draw") return if (idea.draw == "perpetual") "Which check starts it?" else "Which move is it?"
        if (idea.goal == "convert") return "Which trade is it?"
        if (idea.goal == "king") return "Which square does it take from theirs?"
        if (idea.goal == "breakthrough") return "Which pawn goes first?"
        if (idea.goal == "attack") {
            return if (idea.sacrifices.isNotEmpty()) "Which sacrifice opens it?" else "Which check comes first?"
        }
        if (idea.goal == "promote") return "Which move clears its way?"
        if (idea.goal == "defend") return "Which move stops it and keeps your pieces working?"
        if (name != null && name in NUDGE) return NUDGE.getValue(name)
        if (idea.goal == "win") return "Which move wins it?"
        return null
    }

    /**
     * The target on the board, grounded: the focus rung. Squares of their pieces and their
     * king; the player's piece is said separately ([piece]).
     */
    fun target(idea: Idea, board: Board): String? {
        if ((idea.goal == "mate" || idea.goal == "attack") && idea.target != null) {
            val free = idea.flight.map { squareName(it) }
            val room = when {
                free.isEmpty() -> "with no free square"
                free.size == 1 -> "whose only free square is ${free[0]}"
                else -> "with free squares on " + free.dropLast(1).joinToString(", ") + " and ${free.last()}"
            }
            // How many checks there are, while the number narrows the search.
            val checks = if (idea.checks in 1..4) {
                "; you have ${NUMBER[idea.checks] ?: idea.checks.toString()} check" +
                    if (idea.checks != 1) "s" else ""
            } else {
                ""
            }
            return "their king on ${sq(idea.target)}, $room$checks"
        }
        if (idea.goal == "defend" && idea.threat != null) {
            // In words, never as a move: below the move rung no move is named, theirs included.
            val mover = board.pieceAt(idea.threat.from)
            val taken = board.pieceAt(idea.threat.to)
            val who = if (mover != null) "their ${named(board, idea.threat.from)}" else "their threat"
            if (idea.threatened == "mate") return "$who, which threatens mate"
            if (taken != null && taken.colour == board.turn) {
                return "$who, which threatens to take your ${named(board, idea.threat.to)}"
            }
            return "$who, which threatens to win material"
        }
        if (idea.goal == "promote" && idea.first != null) {
            val pawn = queeningPawn(idea, board)
            if (board.pieceTypeAt(pawn) != PAWN) return null
            val rank = squareRank(pawn)
            val toGo = if (board.turn == WHITE) 7 - rank else rank
            val away = "${NUMBER[toGo] ?: toGo.toString()} square${if (toGo != 1) "s" else ""}"
            var text = "your pawn on ${sq(pawn)}, $away from queening"
            if (idea.blow != null && idea.blowBoard != null) {
                val held = idea.blow.squares.drop(1).map { parseSquare(it) }.filter { s ->
                    val p = idea.blowBoard.pieceAt(s)
                    p != null && p.colour != board.turn && p.pieceType != KING
                }
                if (held.isNotEmpty()) text += "; their ${now(idea, board, held[0])} is the piece to deal with"
            }
            return text
        }
        val blow = idea.blow
        val where = idea.blowBoard
        if (blow != null && where != null) {
            val squares = blow.squares.map { parseSquare(it) }
            val theirs = squares.filter { s ->
                val p = where.pieceAt(s)
                p != null && p.colour != board.turn
            }
            val later = if (idea.blowAt > 1) " after your first move" else ""
            if (blow.name == "fork" && idea.blowMove != null) {
                // The first square is where the forking piece lands; the targets follow, and they
                // stand where they are before the fork as after it.
                val targets = squares.drop(1).filter { s ->
                    val p = where.pieceAt(s)
                    p != null && p.colour != board.turn
                }
                if (targets.size >= 2) {
                    val listed = targets.take(3).map { "their ${now(idea, board, it)}" }
                    val namedText = if (listed.size == 1) {
                        listed[0]
                    } else {
                        listed.dropLast(1).joinToString(", ") + " and ${listed.last()}"
                    }
                    val both = if (listed.size == 2) "both" else "all of them"
                    return "$namedText: one move can hit $both$later"
                }
            }
            if (blow.name == "skewer" && squares.size >= 3) {
                return "their ${now(idea, board, squares[1])} and their " +
                    "${now(idea, board, squares[2])}: one line can hold both$later"
            }
            // Every piece named where it stands now, and when the pattern comes about.
            if ((blow.name == "pin" || blow.name == "absolutePin") && squares.size >= 3) {
                return "their ${now(idea, board, squares[1])} and their " +
                    "${now(idea, board, squares[2])} behind it: one line can pin them$later"
            }
            if (blow.name == "discoveredAttack" && squares.size >= 3) {
                return "their ${now(idea, board, squares[2])}, on a line one of your pieces is " +
                    "blocking$later"
            }
            if ((blow.name == "trappedPiece" || blow.name == "hangingPiece") && theirs.isNotEmpty()) {
                val state = if (blow.name == "trappedPiece") "it has no safe square" else "nothing defends it"
                val whenText = if (idea.blowAt > 1) "after your first move, " else ""
                return "their ${now(idea, board, theirs[0])}: $whenText$state"
            }
        }
        val targetPiece = if (idea.target != null) board.pieceAt(idea.target) else null
        if (idea.target != null && targetPiece != null && targetPiece.colour != board.turn) {
            return "their ${named(board, idea.target)}"
        }
        return null
    }

    /** The player's piece that starts it, and the one that strikes when that differs. */
    fun piece(idea: Idea, board: Board): String? {
        val first = idea.first ?: return null
        val moving = board.pieceAt(first.from) ?: return null
        // With its square: "your knight" is two pieces on most boards.
        var text = "your ${PIECE_NAMES.getValue(moving.pieceType)} on ${sq(first.from)}"
        val givenFirst = idea.sacrifices.isNotEmpty() && idea.sacrifices[0].first == 0
        if (givenFirst && idea.sacrifices[0].second in listOf(PIECE_NAMES.getValue(moving.pieceType), "exchange")) {
            text += " -- the piece you give up"
        }
        if (idea.blowMove != null && idea.blowBoard != null && idea.blowAt > 1) {
            val striker = idea.blowBoard.pieceAt(idea.blowMove.from)
            if (striker != null && striker.pieceType != moving.pieceType) {
                text += (if (givenFirst) ", then" else " first, then") +
                    " your ${PIECE_NAMES.getValue(striker.pieceType)}"
            }
        }
        return text
    }

    /**
     * The move played out, correct by construction: the line to the mate, or to the blow and
     * what it wins, the first move named as the sacrifice it is. `describe` is [describeTactic].
     */
    fun why(idea: Idea, board: Board, describe: Describe): String? {
        var text = whyText(idea, board, describe)
        if (text != null && idea.pinnedGuard != null && idea.blowBoard != null) {
            val guard = idea.blowBoard.pieceAt(idea.pinnedGuard)
            if (guard != null) {
                text += "; their ${PIECE_NAMES.getValue(guard.pieceType)} on ${sq(idea.pinnedGuard)} is pinned " +
                    "and cannot take"
            }
        }
        if (text != null && idea.first != null && board.isEnPassant(idea.first)) {
            val first = board.san(idea.first)
            if (text.startsWith(first)) text = "$first, en passant${text.substring(first.length)}"
        }
        if (text == null || idea.first == null || idea.sacrifices.isEmpty() || idea.sacrifices[0].first != 0) {
            return text
        }
        val first = board.san(idea.first)
        val kind = idea.sacrifices[0].second
        val sac = "${if (kind == "exchange") "an" else "a"} $kind sacrifice"
        val rest = (if (text.startsWith(first)) text.substring(first.length) else null) ?: return text
        if (rest.startsWith(":")) return "$first, $sac:${rest.substring(1)}"
        if (rest.startsWith(",")) return "$first, $sac$rest"
        return "$first, $sac,$rest"
    }

    /** `idea_words._why`. */
    private fun whyText(idea: Idea, board: Board, describe: Describe): String? {
        val firstMove = idea.first ?: return null
        val first = board.san(firstMove)
        if (idea.goal == "mate" && idea.mateIn != null && idea.mateIn != 0) {
            val plies = idea.line.take(2 * idea.mateIn - 1).map { it.uci() }
            if (idea.mateIn == 1) return "$first is mate"
            val after = board.copy(withStack = false)
            after.push(firstMove)
            if (idea.mateIn == 2 && plies.size == 3) {
                val reply = after.san(idea.line[1])
                after.push(idea.line[1])
                return "$first, and after $reply, ${after.san(idea.line[2]).trimEnd('#')} is mate"
            }
            return "$first forces mate: $first ${line(after, plies.drop(1))}"
        }
        if (idea.goal == "win" || idea.goal == "combination") {
            var sentence: String? = null
            if (idea.blow != null && idea.blowBoard != null) {
                sentence = describe.describe(idea.blowBoard, idea.blow.name, idea.blow.squares, idea.blowMove)
            }
            var won = if (!idea.won.isNullOrEmpty()) "you win ${idea.won}" else "you win material"
            if (idea.goal == "combination") {
                won = "the material comes back" +
                    if (idea.winning) ", and the position is won" else ", with the better game"
            }
            if (idea.blowAt > 1 && idea.blowMove != null && idea.blowBoard != null) {
                // The moves between the first and the blow, then the blow, then what it does.
                val after = board.copy(withStack = false)
                after.push(firstMove)
                val between = slice(idea.line, 1, 2 * (idea.blowAt - 1)).map { it.uci() }
                val blow = idea.blowBoard.san(idea.blowMove)
                val text = "$first, and after ${line(after, between)}, $blow"
                return text + (if (!sentence.isNullOrEmpty()) ", and $sentence" else "") + "; $won"
            }
            return "$first: " + if (!sentence.isNullOrEmpty()) "$sentence; $won" else won
        }
        if (idea.goal == "promote") {
            val pawn = queeningPawn(idea, board)
            val which = if (pawn == firstMove.from || board.pieceTypeAt(pawn) != PAWN) {
                "the pawn"
            } else {
                "your pawn on ${sq(pawn)}"
            }
            val queens = "$which goes through to queen"
            var sentence: String? = null
            if (idea.blow != null && idea.blowBoard != null) {
                sentence = describe.describe(idea.blowBoard, idea.blow.name, idea.blow.squares, idea.blowMove)
            }
            if (!sentence.isNullOrEmpty() && idea.blowAt > 1 && idea.blowMove != null) {
                val after = board.copy(withStack = false)
                after.push(firstMove)
                val between = slice(idea.line, 1, 2 * (idea.blowAt - 1)).map { it.uci() }
                val blow = idea.blowBoard!!.san(idea.blowMove)
                return "$first, and after ${line(after, between)}, $blow, and $sentence; $queens"
            }
            if (!sentence.isNullOrEmpty()) return "$first: $sentence; $queens"
            return "$first: $queens"
        }
        if (idea.goal == "defend" && idea.threat != null) {
            val passed = board.copy(withStack = false)
            passed.push(Move.NULL)
            return "$first stops ${passed.san(idea.threat)}"
        }
        if (idea.goal == "convert") {
            val shown = idea.line.take(4).map { it.uci() }
            return "$first starts it: ${line(board, shown)}, and the ${idea.ending} ending is won"
        }
        if (idea.goal == "king") {
            if (idea.opposition) return "$first takes the opposition: their king has to give way"
            return "$first: the king leads, and the rest follows"
        }
        if (idea.goal == "breakthrough" && idea.runner != null) {
            val shown = idea.line.take(5).map { it.uci() }
            return "$first breaks through: ${line(board, shown)}, and a pawn runs through"
        }
        if (idea.goal == "draw") {
            val shown = idea.line.take(6).map { it.uci() }
            val how = when (idea.draw ?: "") {
                "perpetual" -> "perpetual check"
                "stalemate" -> "stalemate"
                else -> "a repetition"
            }
            return "$first holds it: ${line(board, shown)}, and it is a draw by $how"
        }
        if (idea.goal == "attack") {
            val shown = idea.line.take(7).map { it.uci() }
            return "$first starts it: ${line(board, shown)}, and their king is in the open"
        }
        return null
    }

    /** Python's `seq[start:end]`, clamped to the list. */
    private fun <T> slice(items: List<T>, start: Int, end: Int): List<T> {
        val from = minOf(start, items.size)
        val to = minOf(maxOf(end, from), items.size)
        return items.subList(from, to)
    }

    /**
     * Where the pawn that queens in the line stands now -- or the first of the player's pawns to
     * reach the seventh -- followed back to the board in front of the player. The first move's
     * square only when that is a pawn: a king's walk that lets a pawn run was read as "your pawn
     * on f5", of the king (Lichess 00g9R).
     */
    private fun queeningPawn(idea: Idea, board: Board): Int {
        val first = idea.first!!
        val us = board.turn
        val seventh = if (us == WHITE) 6 else 1
        val boards = arrayListOf(board)
        var reached: Int? = null
        for ((j, move) in idea.line.withIndex()) {
            val mover = boards.last().pieceAt(move.from)
            val ours = j % 2 == 0 && mover != null && mover.pieceType == PAWN
            if (ours && move.promotion != 0) {
                val start = IdeaReader.origin(board, boards, j, move.from)
                if (start != null) return start
                break
            }
            if (ours && reached == null && squareRank(move.to) == seventh) {
                reached = IdeaReader.origin(board, boards, j, move.from)
            }
            val after = boards.last().copy(withStack = false)
            after.push(move)
            boards.add(after)
        }
        if (reached != null) return reached
        return first.from
    }

    /**
     * Their piece standing on `square` when the blow comes, named where it stands now: the focus
     * hint is read on the board in front of the player.
     */
    private fun now(idea: Idea, board: Board, square: Int): String {
        val blowBoard = idea.blowBoard!!
        val ply = 2 * (idea.blowAt - 1)
        val boards = arrayListOf(board)
        for (move in idea.line.take(ply)) {
            val after = boards.last().copy(withStack = false)
            after.push(move)
            boards.add(after)
        }
        val start = IdeaReader.origin(board, boards, ply, square)
        val blown = blowBoard.pieceAt(square)
        val piece = if (start != null) board.pieceAt(start) else null
        val same = piece != null && blown != null && blown.pieceType == piece.pieceType
        if (start != null && same) return named(board, start)
        // Not on the board yet (a pawn that queens on the way): its name, and no square.
        return if (blown != null) PIECE_NAMES.getValue(blown.pieceType) else "piece"
    }

    private fun line(board: Board, plies: List<String>): String {
        val text = sanLine(board, plies, limit = plies.size)
        return text.split(Regex("\\s+")).filter { it.isNotEmpty() && !it.endsWith(".") && !it.endsWith("...") }
            .joinToString(" ")
    }
}
