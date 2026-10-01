// Analysis (docs/game-review-plan.md, F): the player's own moves from any position of a reviewed
// game, kept as a tree -- the game's moves its main line, the player's lines off it -- the
// engine's best lines for the position on the board, and what the player's own games went on
// with from it (ui/review_analysis.py). A move is checked and named by the sidecar; every
// position the board goes to is asked of the engine and the explorer, the last asked answering
// (a reply for a position the board has left is dropped). What fails is said (`error`).
import type { AnalysisPosition, EngineLines, Explored, PgnTree, ReviewMove } from "$lib/api";

export type AnalysisApi = {
  position: (fen: string) => Promise<{ ok: boolean; error?: string } & AnalysisPosition>;
  move: (fen: string, uci: string) => Promise<{ ok: boolean; error?: string } & AnalysisPosition & { uci: string; san: string }>;
  lines: (fen: string) => Promise<{ ok: boolean; error?: string } & EngineLines>;
  explore: (fen: string, moves: string[]) => Promise<{ ok: boolean; error?: string } & Explored>;
};

/** A move of the tree: the game's own (`game`, with its review's label) or the player's. The
 *  root, the game's first position, has no move. */
export type Node = {
  id: number;
  parent: number | null;
  uci: string | null;
  san: string;
  fen: string;
  ply: number;
  children: number[];
  game: boolean;
  label: string | null;
};

const NO_ANSWER = "Sidecoach is not answering. Try again in a moment.";
const LINES_WAIT_MS = 250;

export class AnalysisDesk {
  nodes = $state<Node[]>([]);
  at = $state(0);
  /** The position on the board as the sidecar read it: its legal moves; null till it answers. */
  here = $state<AnalysisPosition | null>(null);
  /** The engine's lines as last answered: for the board's position only when `fen` is its. */
  lines = $state<EngineLines | null>(null);
  thinking = $state(false);
  engineOn = $state(true);
  explored = $state<Explored | null>(null);
  error = $state("");
  redraw = $state(0);
  private asked = 0;

  constructor(
    private api: AnalysisApi,
    private sound: (fen: string, uci: string) => void,
    /** The room opened again when the sidecar has let it go (idle, or started again): whether it is. */
    private reopen: () => Promise<boolean> = async () => false,
  ) {}

  /** An answer from the room, asked once more after opening it again when it had gone. */
  private async room<T extends { ok: boolean; error?: string }>(ask: () => Promise<T>): Promise<T> {
    const got = await ask();
    if (!got.ok && /not open/.test(got.error ?? "") && (await this.reopen())) return ask();
    return got;
  }

  /** The tree begun again from the game: its first position and every move of it. */
  load(startFen: string, moves: ReviewMove[]): void {
    const nodes: Node[] = [{ id: 0, parent: null, uci: null, san: "", fen: startFen, ply: 0, children: [], game: true, label: null }];
    for (const m of moves) {
      const id = nodes.length;
      nodes[id - 1].children.push(id);
      nodes.push({ id, parent: id - 1, uci: m.uci, san: m.san, fen: m.fen_after, ply: m.ply, children: [], game: true, label: m.label });
    }
    this.nodes = nodes;
    this.at = 0;
  }

  get node(): Node | null {
    return this.nodes[this.at] ?? null;
  }

  /** The moves from the game's start to the board's position, UCI. */
  path(id = this.at): string[] {
    const out: string[] = [];
    for (let n: Node | undefined = this.nodes[id]; n && n.uci; n = n.parent != null ? this.nodes[n.parent] : undefined) out.unshift(n.uci);
    return out;
  }

  /** The node of the game's move `ply` (0: its first position). */
  gameNode(ply: number): number {
    let id = 0;
    for (let i = 0; i < ply; i++) {
      const next = this.nodes[id]?.children.find((c) => this.nodes[c].game);
      if (next == null) break;
      id = next;
    }
    return id;
  }

  /** To a move of the tree: the board goes there (a step on is heard), and is read again. */
  go(id: number, heard = false): void {
    const to = this.nodes[id];
    if (!to) return;
    if (heard && to.parent === this.at && to.uci) this.sound(this.nodes[this.at].fen, to.uci);
    this.at = id;
    void this.read();
  }

  /** Back to the move before, or on along the line: the game's move where it goes on, else the
   *  line the player played first from here. */
  step(delta: number): void {
    const n = this.node;
    if (!n) return;
    if (delta < 0 && n.parent != null) this.go(n.parent);
    else if (delta > 0 && n.children.length) this.go(n.children[0], true);
  }

  /** To the line's start or end: the root, or as far as the first moves go from here. */
  toEnd(last: boolean): void {
    if (!last) return this.go(0);
    let id = this.at;
    while (this.nodes[id].children.length) id = this.nodes[id].children[0];
    this.go(id);
  }

  /** A move on the board (the hand put it there): the line's own when it was played from here
   *  before, else checked and named by the sidecar and added as a line of the player's. */
  play = async (uci: string): Promise<void> => {
    const from = this.node;
    if (!from) return;
    const known = from.children.find((c) => this.nodes[c].uci === uci);
    if (known != null) {
      this.go(known, true);
      return;
    }
    this.sound(from.fen, uci);
    const played = await this.api.move(from.fen, uci).catch(() => null);
    if (!played || !played.ok) {
      this.error = played?.error || NO_ANSWER;
      this.redraw++;
      return;
    }
    this.error = "";
    const id = this.add(from.id, played.uci, played.san, played.fen);
    this.at = id;
    this.here = { fen: played.fen, side: played.side, legal: played.legal, over: played.over };
    void this.read(false);
  };

  /** A line of moves played on from the board's position, as far as it goes (an engine's line
   *  to its `upto`th move): each the tree's where it was played before. */
  async playLine(steps: { uci: string; san: string; fen: string }[], upto: number): Promise<void> {
    let id = this.at;
    for (const s of steps.slice(0, upto + 1)) {
      const known = this.nodes[id].children.find((c) => this.nodes[c].uci === s.uci);
      id = known ?? this.add(id, s.uci, s.san, s.fen);
    }
    this.go(id);
  }

  /** A line of the player's taken out, with what follows it; the board back to where it began
   *  when it was on it. The game's own moves stay. */
  remove(id: number): void {
    const n = this.nodes[id];
    if (!n || n.game || n.parent == null) return;
    const gone = new Set<number>();
    const walk = (x: number) => { gone.add(x); for (const c of this.nodes[x].children) walk(c); };
    walk(id);
    const parent = n.parent;
    let back = this.at;
    while (gone.has(back)) back = this.nodes[back].parent ?? 0;
    // Ids are places in the list: the removed stay as empty places, never reached again.
    this.nodes = this.nodes.map((x) =>
      x.id === parent ? { ...x, children: x.children.filter((c) => c !== id) } : gone.has(x.id) ? { ...x, parent: null, children: [] } : x,
    );
    if (back !== this.at) this.go(back);
  }

  /** The tree as the PGN takes it: from the game's start, the game's own move first at each. */
  tree(id = 0): PgnTree[] {
    return this.nodes[id].children.map((c) => ({ uci: this.nodes[c].uci ?? "", children: this.tree(c) }));
  }

  setEngine(on: boolean): void {
    this.engineOn = on;
    if (!on) this.lines = null;
    else void this.read(false);
  }

  private add(parent: number, uci: string, san: string, fen: string): number {
    const id = this.nodes.length;
    const from = this.nodes[parent];
    this.nodes = [
      ...this.nodes.map((x) => (x.id === parent ? { ...x, children: [...x.children, id] } : x)),
      { id, parent, uci, san, fen, ply: from.ply + 1, children: [], game: false, label: null },
    ];
    return id;
  }

  /** The board's position asked of the sidecar: its legal moves (unless known), the engine's
   *  lines, the explorer. Each answer is kept only if the board is still there. */
  private async read(position = true): Promise<void> {
    const n = this.node;
    if (!n) return;
    const ask = ++this.asked;
    const fen = n.fen;
    const still = () => ask === this.asked;
    if (position) {
      this.here = null;
      void this.api.position(fen).then((p) => {
        if (!still()) return;
        if (p.ok) this.here = p;
        else this.error = p.error || NO_ANSWER;
      }).catch(() => still() && (this.error = NO_ANSWER));
    }
    void this.room(() => this.api.explore(fen, this.path(n.id))).then((e) => {
      if (still() && e.ok) this.explored = e;
    }).catch(() => {});
    if (!this.engineOn) return;
    // The engine asked once the board stays: stepping through quickly searches only where it stops.
    this.thinking = true;
    await new Promise((r) => setTimeout(r, LINES_WAIT_MS));
    if (!still()) return;
    const got = await this.room(() => this.api.lines(fen)).catch(() => null);
    if (!still()) return;
    this.thinking = false;
    if (got?.ok) this.lines = got;
    else {
      this.lines = null;
      this.error = got?.error || NO_ANSWER;
    }
  }
}
