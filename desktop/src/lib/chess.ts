// Small pure helpers the pages share: the FEN's placement, square geometry for either
// orientation, sprite names, and the formatting of the status strip's figures.

export const FILES = "abcdefgh";

/** 'K' -> 'wK', 'n' -> 'bN': the sprite class of a FEN piece letter. */
export const spriteOf = (ch: string): string => (ch === ch.toUpperCase() ? "w" : "b") + ch.toUpperCase();
export const colourOf = (ch: string): "white" | "black" => (ch === ch.toUpperCase() ? "white" : "black");

/** "e2e4" -> the two squares it touches, for the last-move highlight. */
export function moveSquares(uci: string | null | undefined): Set<string> {
  if (!uci || uci.length < 4) return new Set();
  return new Set([uci.slice(0, 2), uci.slice(2, 4)]);
}

/** square -> piece letter, from the FEN's first field. */
export function placement(fen: string): Map<string, string> {
  const out = new Map<string, string>();
  const rows = fen.split(" ")[0].split("/");
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r] ?? "") {
      if (ch >= "1" && ch <= "8") { file += +ch; continue; }
      out.set(FILES[file] + (8 - r), ch);
      file++;
    }
  }
  return out;
}

/** The FEN's ranks, rank 8 first, each expanded to 8 cells (null = empty). */
export function ranksOf(fen: string): (string | null)[][] {
  return fen.split(" ")[0].split("/").map((row) => {
    const cells: (string | null)[] = [];
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") for (let i = 0; i < +ch; i++) cells.push(null);
      else cells.push(ch);
    }
    return cells;
  });
}

/** Display column/row (0..7 from the left/top) of a square for an orientation. */
export function cellOf(square: string, bottom: string): { x: number; y: number } {
  const file = FILES.indexOf(square[0]), rank = +square[1] - 1;
  const col = bottom === "black" ? 7 - file : file;
  const row = bottom === "black" ? rank : 7 - rank;
  return { x: col, y: row };
}

/** Display column/row -> square name, or null off the board. */
export function squareAt(x: number, y: number, bottom: string): string | null {
  const file = bottom === "black" ? 7 - x : x;
  const rank = bottom === "black" ? y : 7 - y;
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
  return FILES[file] + (rank + 1);
}

/** The squares and moves a coach message names, so hovering one can light its square. */
export const SQ_TOKEN = /(?<![A-Za-z0-9])((?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?|[a-h][1-8])(?![A-Za-z0-9])/g;

export const motifWords = (m: string | null | undefined): string => (m || "").replace(/([a-z])([A-Z0-9])/g, "$1 $2").toLowerCase();

export const kilo = (n: number): string => (n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k" : String(n));

/** mm:ss.s from seconds. */
export function stamp(t: number): string {
  const m = Math.floor(t / 60), s = (t - m * 60).toFixed(1).padStart(4, "0");
  return String(m).padStart(2, "0") + ":" + s;
}

export function clockText(clock: { white_ms: number | null; black_ms: number | null; running: string | null } | null | undefined): string {
  if (!clock) return "–";
  const fmt = (ms: number | null) => { if (ms == null) return "–"; const s = Math.round(ms / 1000); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
  const run = clock.running ? " (" + clock.running + " running)" : "";
  return "W " + fmt(clock.white_ms) + "  B " + fmt(clock.black_ms) + run;
}

/** Latency, amortised: "1.4 s (last 1.1, n=12)". */
export function latencyText(mean: number | null | undefined, last: number | null | undefined, n: number | undefined): string {
  if (mean == null || last == null) return "–";
  return mean.toFixed(1) + " s (last " + last.toFixed(1) + ", n=" + n + ")";
}

/** "$0.023" or "$1.20". */
export function money(cost: number): string {
  return "$" + (cost < 0.01 ? cost.toFixed(4) : cost < 0.1 ? cost.toFixed(3) : cost.toFixed(2));
}
