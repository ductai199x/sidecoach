// The pieces on a layer of their own, each placed by a transform, so a move slides and a
// taken piece fades: the boards that play moves (the puzzle page, the review) share it.
import { cellOf, placement, spriteOf } from "./chess";

// A taken piece's fade (panel.css .piece.fading: it waits for the capturer to arrive), and then
// its node goes.
const FADE_MS = 240;

export class PieceLayer {
  /** square -> the piece's node */
  nodes = new Map<string, HTMLDivElement>();

  constructor(
    private el: HTMLDivElement,
    public bottom: string,
  ) {}

  place(node: HTMLDivElement, square: string): void {
    const { x, y } = cellOf(square, this.bottom);
    node.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
  }

  /** Put down without sliding: a dropped piece lands where the pointer let it go. */
  placeNow(node: HTMLDivElement, square: string): void {
    node.style.transition = "none";
    this.place(node, square);
    node.getBoundingClientRect(); // settle before the transition returns
    node.style.transition = "";
  }

  /** The pieces as `fen` has them; `uci`, the move that made it, names the piece that moves.
   *  Returns square -> letter, what is on the board now. */
  render(fen: string, uci: string | null = null): Record<string, string> {
    const want = placement(fen);
    const stay = new Map<string, HTMLDivElement>();
    const gone: [string, string, HTMLDivElement][] = [];
    for (const [sq, node] of this.nodes) {
      if (want.get(sq) === node.dataset.letter) {
        // Placed again even so: a piece shown ahead of the position (a move sent, a wrong
        // one shown, the board turned) goes where the position has it.
        this.place(node, sq);
        stay.set(sq, node);
      } else gone.push([sq, node.dataset.letter!, node]);
    }
    const next = new Map(stay);
    for (const [sq, letter] of want) {
      if (stay.has(sq)) continue;
      // The move's piece slides here from its square; any other piece of this kind that left
      // a square (a castling rook) slides here too; otherwise it appears.
      let i = uci && sq === uci.slice(2, 4) ? gone.findIndex(([from, l]) => from === uci.slice(0, 2) && l === letter) : -1;
      if (i < 0) i = gone.findIndex(([, l]) => l === letter);
      if (i >= 0) {
        const [[, , node]] = gone.splice(i, 1);
        this.place(node, sq);
        next.set(sq, node);
      } else { // a piece back on the board (a move taken back over a capture): fades in
        const node = document.createElement("div");
        node.className = "piece appearing " + spriteOf(letter);
        node.dataset.letter = letter;
        this.placeNow(node, sq);
        this.el.appendChild(node);
        requestAnimationFrame(() => node.classList.remove("appearing"));
        next.set(sq, node);
      }
    }
    for (const [, , node] of gone) { // taken, or a promoted pawn: fades under the piece arriving
      node.classList.add("fading");
      setTimeout(() => node.remove(), FADE_MS);
    }
    this.nodes = next;
    const occupied: Record<string, string> = {};
    for (const [sq, node] of this.nodes) occupied[sq] = node.dataset.letter!;
    return occupied;
  }

  /** Turned round: every piece to its square seen from the other side, sliding. */
  turn(bottom: string): void {
    this.bottom = bottom;
    for (const [sq, node] of this.nodes) this.place(node, sq);
  }
}
