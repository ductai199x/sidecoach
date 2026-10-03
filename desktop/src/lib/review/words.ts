// What the review says before the coach's own words are ready: a headline per kind of move,
// a sentence from the facts the review computed (the better move, what the move changed, the
// motifs), the eval as chess.com writes it, and the summary's one line on the game. The
// coach's prepared words (docs/game-review-plan.md, phase 4) take their place when present.
import type { Eval, GameSummary, ReviewMove } from "$lib/api";
import { motifWords } from "$lib/chess";

export const CLASSES = ["brilliant", "great", "best", "excellent", "good", "book", "inaccuracy", "mistake", "miss", "blunder"] as const;
/** The rows the summary shows folded, as chess.com folds its table. */
export const FOLDED = ["brilliant", "great", "best", "mistake", "miss", "blunder"];
export const BAD = new Set(["inaccuracy", "mistake", "miss", "blunder"]);
export const KEY = new Set(["brilliant", "great", "inaccuracy", "mistake", "miss", "blunder"]);

export const className = (c: string): string => c[0].toUpperCase() + c.slice(1);

/** An opening's name without its ECO code: "B00 King's Pawn Game" -> "King's Pawn Game". */
export const openingName = (o: string): string => o.replace(/^[A-E]\d\d\s+/, "");

/** The motifs not worth naming on a move: what its SAN already says (a capture, a check,
 *  castling), or nothing the player can use (a quiet move). */
const UNSAID = new Set(["quietMove", "capture", "check", "castling"]);

/** A move's number in a move list, from the FEN it is played from: "23." for White's, "23..."
 *  for Black's. Never from the ply: a game set up from a position starts at its own number, with
 *  either side to move (the review of 2026-10-01). */
export function moveNumber(fen: string): { n: number; white: boolean } {
  const fields = fen.split(" ");
  return { n: +fields[5] || 1, white: fields[1] !== "b" };
}
export function moveLabel(fen: string): string {
  const { n, white } = moveNumber(fen);
  return `${n}${white ? "." : "..."}`;
}

/** A line's moves as a move list writes them: "11. Qd3 h6 12. O-O", or "11... h6" when
 *  Black moves first; the numbers from the FEN the line starts from. */
export function numbered(from: string, sans: string[]): string[] {
  const fields = from.split(" ");
  let white = fields[1] !== "b";
  let n = +fields[5] || 1;
  return sans.map((san, i) => {
    const text = white ? `${n}. ${san}` : i === 0 ? `${n}... ${san}` : san;
    if (!white) n += 1;
    white = !white;
    return text;
  });
}

const IS: Record<string, string> = {
  brilliant: "is brilliant",
  great: "is a great move",
  best: "is best",
  excellent: "is excellent",
  good: "is good",
  book: "is a book move",
  inaccuracy: "is an inaccuracy",
  mistake: "is a mistake",
  miss: "is a miss",
  blunder: "is a blunder",
};

export function headline(m: ReviewMove): string {
  return `${m.san} ${IS[m.label ?? ""] ?? "was played"}`;
}

/** A retried move that solves the position: it stays on the board. */
export const SOLVED = new Set(["brilliant", "great", "best", "excellent"]);

/** What a retried move earns, in chess.com's manner: a move that solves it congratulated. */
export function tryWords(t: { san: string; label: string | null; only: boolean }): string {
  const label = t.label ?? "";
  if (label === "best" || label === "great" || label === "brilliant") {
    return t.only ? `Well found! ${t.san} is the only move that works.` : `Well found! ${t.san} ${IS[label]}.`;
  }
  if (label === "excellent") return `Good find! ${t.san} is excellent, nearly the best.`;
  // An invitation to look again, never a verdict and an order (Tai, 2026-10-02).
  if (label === "good") return `${t.san} is good, and there's something better. Have another look.`;
  if (BAD.has(label)) return t.only ? `${t.san} ${IS[label]} — only one move works here. Have another look.` : `${t.san} ${IS[label]}. Have another look.`;
  return `${t.san}. Have another look.`;
}

/** The eval as chess.com writes it, from White's side: +1.24, -0.3, M3, -M2, 1-0. */
export function evalText(e: Eval | null | undefined): string {
  if (!e) return "";
  if (e.result) return e.result.replaceAll("1/2", "½");
  if (e.mate != null) return (e.mate < 0 ? "-M" : "M") + Math.abs(e.mate);
  const pawns = (e.cp ?? 0) / 100;
  return (pawns > 0 ? "+" : "") + (Math.abs(pawns) >= 10 ? pawns.toFixed(0) : pawns.toFixed(2).replace(/\.?0+$/, "") || "0");
}

/** Whose side the eval favours, for the pill's colour. */
export function evalSide(e: Eval | null | undefined): "w" | "b" {
  if (!e) return "w";
  if (e.result) return e.result === "0-1" ? "b" : "w";
  if (e.mate != null) return e.mate < 0 ? "b" : "w";
  return (e.cp ?? 0) < 0 ? "b" : "w";
}

/** The mover's expected score before and after the move, from White's scores. */
function moverScores(m: ReviewMove, whiteBefore: number): [number, number] {
  return m.side === "white" ? [whiteBefore, m.white_score] : [1 - whiteBefore, 1 - m.white_score];
}

function standing(score: number): string {
  if (score >= 0.9) return "winning";
  if (score >= 0.7) return "clearly better";
  if (score >= 0.57) return "a little better";
  if (score > 0.43) return "about equal";
  if (score > 0.3) return "a little worse";
  if (score > 0.1) return "clearly worse";
  return "lost";
}

/** A sentence or two on the move, from what the review knows. */
// `reveal`: whether a move the player could retry names the better one. Off while the best
// move's arrow is off, as the arrow would: Retry asks the player to find it, Best shows it.
// `lastBook`: the game's last book move, a key moment of its own.
export function sentence(m: ReviewMove, whiteBefore: number, reveal: boolean, lastBook = false): string {
  const [before, after] = moverScores(m, whiteBefore);
  const label = m.label ?? "";
  const motifs = m.tags.filter((t) => !UNSAID.has(t)).map(motifWords).filter(Boolean);
  const better = m.best_san && m.best_san !== m.san ? m.best_san : null;
  const parts: string[] = [];
  if (label === "book" && lastBook) {
    return `${m.opening ? `The last book move of the ${openingName(m.opening)}.` : "The last book move."} From here on, the game is its own.`;
  }
  if (label === "book") return m.opening ? `A known move of the ${openingName(m.opening)}.` : "A known opening move.";
  if (label === "brilliant") parts.push("A sacrifice that works: the engine's own choice, and not an easy one to see.");
  else if (label === "great") parts.push("The one move that holds everything together here.");
  else if (label === "best") parts.push("The engine's first choice.");
  else if (label === "excellent" || label === "good") parts.push(better ? `Close to the best, ${better}.` : "A sound move.");
  else if (label === "miss") parts.push(better && reveal ? `${better} was there, and it won more.` : "A chance to win more went by.");
  else if (BAD.has(label)) parts.push(better && reveal ? `${better} was best.` : "There was a better move.");
  if (BAD.has(label) && standing(before) !== standing(after)) {
    parts.push(`${m.side === "white" ? "White" : "Black"} was ${standing(before)}; after it, ${standing(after)}.`);
  }
  if (motifs.length) parts.push(`${motifs.length === 1 ? "Theme" : "Themes"}: ${motifs.join(", ")}.`);
  if (BAD.has(label) && better && !reveal) parts.push("Retry to find it, or Best to see it.");
  return parts.join(" ");
}

/** What the best move kept that the move played did not, from the review's scores: said
 *  under "Qd4 was best" when the AI coach has not written on the move. */
export function bestWords(m: ReviewMove, whiteBefore: number): string {
  const [before, after] = moverScores(m, whiteBefore);
  const side = m.side === "white" ? "White" : "Black";
  if (standing(before) !== standing(after)) return `It kept ${side} ${standing(before)}; ${m.san}, played instead, left ${side} ${standing(after)}.`;
  if (m.loss < 0.02) return `${m.san}, played instead, was nearly as good.`;
  return `${m.san}, played instead, gave a little of it away.`;
}

/** The summary's one line on the game, from its shape and result. */
export function verdict(s: GameSummary, result: string | null): string {
  const winner = result === "1-0" ? "White" : result === "0-1" ? "Black" : null;
  const loser = winner === "White" ? "Black" : "White";
  switch (s.shape) {
    case "smooth": return winner ? `${winner} took the lead and never let go.` : "One side led all the way, and it still ended level.";
    case "balanced": return winner ? `Close throughout, until ${winner} found the way through.` : "A balanced game from start to finish.";
    case "sudden": return winner ? `Level for most of the game, then ${winner} broke it open late.` : "Level for most of the game, and a late scare.";
    case "giveaway": return winner ? `${loser} was winning, and ${winner} turned it around.` : "A winning position slipped away.";
    case "wild": return "A wild game: the lead changed hands again and again.";
    case "intense": return "An intense game: the lead changed hands, then changed back.";
    case "sharp": return winner ? `A sharp game with big swings, and ${winner} kept the upper hand.` : "A sharp game with big swings.";
    default: return winner ? `${winner} won.` : result === "1/2-1/2" ? "A draw." : "The game, move by move.";
  }
}
