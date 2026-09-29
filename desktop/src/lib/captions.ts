// The captions' box: the coach's words over the other windows, the way Live Caption shows
// speech -- the phone's `ui/Captions.kt`, on the desktop since Tai asked for it in fast
// games (2026-09-29: "I think it's really useful in fast games"). What is pure is here, for
// the two pages (routes/captions, the box; routes/captions-cross, the cross a box is dropped
// on to turn them off) to share.

/** Lines the words show until the box is sized, when it is double-clicked, and at most: the
 * box is sized by whole lines, so none shows cut in half (the phone's). */
export const LINES = 2;
export const EXPANDED_LINES = 6;
export const MAX_LINES = 10;
/** How near an edge a pointer sizes the box instead of moving it, and the narrowest box, in
 * CSS pixels (Tai, 2026-09-29: "just make the box sizable"). */
export const EDGE = 8;
export const MIN_WIDTH = 240;
/** How long the box stays after the voice stops, as Live Caption's does. */
export const HIDE_AFTER_MS = 5000;
/** How long the moves a click shows stay, after the click or the last touch on them. */
export const CONTROLS_MS = 4000;
/** The words kept, the oldest cut at a space. */
export const KEEP_CHARS = 1200;
/** A click and the next one within this is a double-click, never two clicks. */
export const DOUBLE_CLICK_MS = 300;
/** How far a pointer moves before a press is a drag, in CSS pixels. */
export const SLOP = 4;
/** The cross's centre above the foot of its monitor's work area, and how near it the pointer
 * is let go to turn captions off -- the pointer, not the box: the box's centre near a cross
 * 100 px up made every box parked low a dismissal (the phone's, Tai, 2026-09-29: "the dismiss thing is STILL TOO FAR UP"). */
export const CROSS_LIFT = 40;
export const CROSS_REACH = 44;
/** The sample line a change to the captions shows at once (the phone's `preview`). */
export const PREVIEW = "This is how captions look. Spot on -- rooks belong on open files.";

/** A rectangle in physical screen pixels. */
export type Rect = { left: number; top: number; right: number; bottom: number };

/** Where a box of `w` x `h` at (`x`, `y`) goes so it does not cover the board: above it or
 * below it, whichever is nearer and has the room, else the side with more room. Kept on the
 * screen either way. The vision reads the board from the screen, and a box over it would
 * hide squares. The phone's `Captions.clear`. */
export function clear(
  x: number, y: number, w: number, h: number, screen: Rect, board: Rect | null, margin: number,
): [number, number] {
  const cx = Math.min(Math.max(x, screen.left), Math.max(screen.left, screen.right - w));
  let cy = Math.min(Math.max(y, screen.top), Math.max(screen.top, screen.bottom - h));
  if (board) {
    const guard = { left: board.left - margin, top: board.top - margin, right: board.right + margin, bottom: board.bottom + margin };
    const over = cx < guard.right && cx + w > guard.left && cy < guard.bottom && cy + h > guard.top;
    if (over) {
      const above = guard.top - h;
      const below = guard.bottom;
      const roomAbove = above >= screen.top;
      const roomBelow = below + h <= screen.bottom;
      const middle = cy + h / 2;
      const boardMiddle = (board.top + board.bottom) / 2;
      if (roomAbove && roomBelow) cy = middle < boardMiddle ? above : below;
      else if (roomAbove) cy = above;
      else if (roomBelow) cy = below;
      else if (guard.top - screen.top >= screen.bottom - guard.bottom) cy = screen.top;
      else cy = screen.bottom - h;
    }
  }
  return [cx, cy];
}

/** The words said so far and `text` after them: on a line of its own when it starts a message
 * (Tai, 2026-09-29: "the text do not have any new lines what so ever so it's always really hard
 * to read"), run on when it continues one; the oldest cut at a space past KEEP_CHARS; a dash as
 * the coach writes it (two hyphens) read as a dash. */
export function joined(said: string, text: string, ownLine: boolean): string {
  const line = text.trim().replaceAll(" -- ", " — ");
  let out = said ? said + (ownLine ? "\n" : " ") + line : line;
  if (out.length > KEEP_CHARS) {
    const at = out.indexOf(" ", out.length - KEEP_CHARS);
    out = out.slice(at < 0 ? out.length - KEEP_CHARS : at + 1);
  }
  return out;
}

/** A move or a square as the coach writes it (SAN, castling, a bare square, or a square or a
 * pawn move a sentence starts with a capital: "G4 was a miss"), and a piece named: the words
 * a player looks for, set off in the box (Tai, 2026-09-29: "why are the pieces and their names
 * and square names and SAN annotations not highlighted"). */
const MOVE = String.raw`(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[A-H](?:x[a-h])?[1-8])[+#]?`;
const PIECE = String.raw`(?:[Kk]ing|[Qq]ueen|[Rr]ook|[Bb]ishop|[Kk]night|[Pp]awn)s?`;
const TOKENS = new RegExp(String.raw`(?<![A-Za-z0-9])(?:(${MOVE})|(${PIECE}))(?![A-Za-z0-9])`, "g");

export type Run = { text: string; kind: "move" | "piece" | null };

/** `text` as runs: the moves and squares, the pieces, and the words between. */
export function runs(text: string): Run[] {
  const out: Run[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKENS)) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), kind: null });
    out.push({ text: m[0], kind: m[1] ? "move" : "piece" });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), kind: null });
  return out;
}

/** The colour moves and squares stand out in, against `background`: a warm gold on a dark box,
 * a deep amber on a light one. */
export function accent(background: string): string {
  const hex = /^#?([0-9a-f]{6})$/i.exec(background.trim())?.[1] ?? "1F1F1F";
  const n = parseInt(hex, 16);
  const luma = 0.2126 * ((n >> 16) & 255) + 0.7152 * ((n >> 8) & 255) + 0.0722 * (n & 255);
  return luma < 140 ? "#FFD479" : "#8A5300";
}

/** A colour "#RRGGBB" and an opacity in percent as a CSS colour; a colour that does not read
 * keeps `fallback` (the phone's `captionStyle`). */
export function fill(colour: string, opacityPercent: number, fallback = "#1F1F1F"): string {
  const hex = /^#?([0-9a-f]{6})$/i.exec(colour.trim())?.[1] ?? fallback.slice(1);
  const n = parseInt(hex, 16);
  const alpha = Math.min(100, Math.max(20, opacityPercent)) / 100;
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}
