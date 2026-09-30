// The player's hand on a board that takes moves, as chess.com's: press a piece of theirs and it
// lifts to the pointer, its square lit and its legal squares dotted; drop it on one, or let go
// and click one. A dropped piece lands where it is dropped, a clicked move slides; a pawn
// reaching the last rank offers a picker over the square. A right-click marks a square, a
// right-drag draws an arrow, a left-click clears them. What a move means is the page's
// (`play`): the puzzle page sends it to its room, the review grades a retry.
import { cellOf, colourOf, squareAt } from "./chess";
import type { PieceLayer } from "./pieces";

export type Promo = { square: string; x: number; y: number; down: boolean; resolve: (piece: string | null) => void };

export type Hand = {
  /** The squares' element, whose box is the board's geometry. */
  board: () => HTMLDivElement | null;
  /** The element that captures the pointer while a piece is held. */
  wrap: () => HTMLDivElement | null;
  layer: () => PieceLayer | null;
  bottom: () => string;
  /** Whether the player may move now, and whose pieces are theirs. */
  canMove: () => boolean;
  side: () => string;
  /** From a square, the squares its piece may go to. */
  legal: () => Record<string, string[]>;
  /** A legal move made: its piece is already where it went. */
  play: (uci: string) => void;
};

// The pointer kept by the board while a piece is held; a pointer already gone (released between
// the press and here) cannot be, and the hand goes on without it.
function capture(el: HTMLElement, pointerId: number): void {
  try {
    el.setPointerCapture(pointerId);
  } catch {
    /* gone already */
  }
}

export class MoveInput {
  selected = $state<string | null>(null);
  targets = $state<string[]>([]);
  hover = $state<string | null>(null); // the square under a dragged piece
  marked = $state<string[]>([]);
  arrows = $state<[string, string][]>([]);
  promo = $state<Promo | null>(null);
  private drag: { square: string; node: HTMLDivElement; moved: boolean; startX: number; startY: number } | null = null;
  private rightDrag: { from: string | null } | null = null;

  constructor(private hand: Hand) {}

  clear(): void {
    this.selected = null;
    this.targets = [];
  }

  clearMarks(): void {
    this.marked = [];
    this.arrows = [];
  }

  private squareUnder(clientX: number, clientY: number): string | null {
    const board = this.hand.board();
    if (!board) return null;
    const r = board.getBoundingClientRect();
    return squareAt(Math.floor(((clientX - r.left) / r.width) * 8), Math.floor(((clientY - r.top) / r.height) * 8), this.hand.bottom());
  }

  private mine(square: string): HTMLDivElement | null {
    const node = this.hand.layer()?.nodes.get(square) ?? null;
    return node && this.hand.canMove() && colourOf(node.dataset.letter!) === this.hand.side() ? node : null;
  }

  private select(square: string): void {
    if (!this.mine(square)) { this.clear(); return; }
    this.selected = square;
    this.targets = this.hand.legal()[square] ?? [];
  }

  private promotionPicker(square: string): Promise<string | null> {
    return new Promise((resolve) => {
      const { x, y } = cellOf(square, this.hand.bottom());
      this.promo = { square, x, y, down: y > 3, resolve: (piece) => { this.promo = null; resolve(piece); } };
    });
  }

  // The move: the piece put on its square at once -- where it was dropped, or slid there from a
  // click -- a promotion's piece picked, then handed to the page.
  private async tryMove(from: string, to: string, dropped: boolean): Promise<void> {
    if (!this.targets.includes(to)) return;
    this.clear();
    const layer = this.hand.layer();
    const node = layer?.nodes.get(from) ?? null;
    let uci = from + to;
    if (node && node.dataset.letter!.toUpperCase() === "P" && (to[1] === "8" || to[1] === "1")) {
      layer?.placeNow(node, to);
      const piece = await this.promotionPicker(to);
      if (!piece) { layer?.placeNow(node, from); return; }
      uci += piece;
    } else if (node && layer) {
      if (dropped) layer.placeNow(node, to); else layer.place(node, to);
    }
    this.hand.play(uci);
  }

  // Under the pointer: a piece picked up is centred on it, as on chess.com.
  private follow(node: HTMLDivElement, clientX: number, clientY: number): void {
    const board = this.hand.board();
    if (!board) return;
    const r = board.getBoundingClientRect();
    const cell = r.width / 8;
    const px = clientX - r.left - cell / 2, py = clientY - r.top - cell / 2;
    node.style.transform = `translate(${(px / cell) * 100}%, ${(py / cell) * 100}%)`;
  }

  down = (e: PointerEvent): void => {
    const wrap = this.hand.wrap();
    if (!wrap) return;
    if (e.button === 2) { // a right-click: a mark, or an arrow if dragged
      e.preventDefault();
      this.rightDrag = { from: this.squareUnder(e.clientX, e.clientY) };
      capture(wrap, e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    if (this.marked.length || this.arrows.length) this.clearMarks(); // a left-click clears them
    const square = this.squareUnder(e.clientX, e.clientY);
    if (!square) return;
    if (this.selected && this.targets.includes(square)) { void this.tryMove(this.selected, square, false); return; }
    const node = this.mine(square);
    if (node) {
      // Lifted at the press, centred under the pointer.
      this.drag = { square, node, moved: false, startX: e.clientX, startY: e.clientY };
      capture(wrap, e.pointerId);
      node.classList.add("dragging");
      this.follow(node, e.clientX, e.clientY);
      this.hover = square;
      this.select(square);
    } else if (this.selected) {
      this.clear();
    }
  };

  move = (e: PointerEvent): void => {
    const drag = this.drag;
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) >= 4) {
      drag.moved = true;
      document.body.classList.add("grabbing");
    }
    this.follow(drag.node, e.clientX, e.clientY);
    this.hover = this.squareUnder(e.clientX, e.clientY);
  };

  // A piece put down from a drag: on a legal square it is the move, landed where it was dropped;
  // anywhere else it is back on its square at once, still picked.
  private putDown(drag: { square: string; node: HTMLDivElement }, to: string | null): void {
    this.hover = null;
    document.body.classList.remove("grabbing");
    if (to && to !== drag.square && this.targets.includes(to)) void this.tryMove(drag.square, to, true);
    else this.hand.layer()?.placeNow(drag.node, drag.square);
    drag.node.classList.remove("dragging");
  }

  up = (e: PointerEvent): void => {
    if (this.rightDrag) {
      const to = this.squareUnder(e.clientX, e.clientY);
      const from = this.rightDrag.from;
      this.rightDrag = null;
      if (from && to && from !== to) {
        const i = this.arrows.findIndex(([a, b]) => a === from && b === to);
        this.arrows = i >= 0 ? this.arrows.filter((_, j) => j !== i) : [...this.arrows, [from, to]];
      } else if (from) {
        this.marked = this.marked.includes(from) ? this.marked.filter((m) => m !== from) : [...this.marked, from];
      }
      return;
    }
    const drag = this.drag;
    if (!drag) return;
    this.drag = null;
    this.putDown(drag, drag.moved ? this.squareUnder(e.clientX, e.clientY) : null); // a click: stays picked
  };

  cancel = (): void => {
    if (this.drag) { this.putDown(this.drag, null); this.drag = null; }
    this.rightDrag = null;
  };
}
