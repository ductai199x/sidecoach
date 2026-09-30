// Arrows on a board as chess.com draws them: a broad shaft from the piece's square to a wide
// head on the target, a knight's in an L (its long leg first); in board units (a square is 1),
// for an SVG with the viewBox 0 0 8 8. Drawn as a shaft (a stroke) and a head (a polygon) in one
// group, so a colour laid at partial opacity never darkens where the two meet.
import { cellOf } from "$lib/chess";

const SHAFT = 0.2;
const HEAD_WIDTH = 0.5;
const HEAD_LENGTH = 0.38;

export type Drawn = { shaft: string; head: string; width: number };

export function arrow(from: string, to: string, bottom: string): Drawn {
  const a = cellOf(from, bottom), b = cellOf(to, bottom);
  const start = { x: a.x + 0.5, y: a.y + 0.5 };
  const tip = { x: b.x + 0.5, y: b.y + 0.5 };
  const dx = b.x - a.x, dy = b.y - a.y;
  // A knight's move turns once: along its long leg, then the short one.
  const knight = (Math.abs(dx) === 1 && Math.abs(dy) === 2) || (Math.abs(dx) === 2 && Math.abs(dy) === 1);
  const corner = knight ? (Math.abs(dx) === 2 ? { x: tip.x, y: start.y } : { x: start.x, y: tip.y }) : null;
  const last = corner ?? start; // where the final leg begins
  const len = Math.hypot(tip.x - last.x, tip.y - last.y) || 1;
  const ux = (tip.x - last.x) / len, uy = (tip.y - last.y) / len;
  // The shaft stops where the head begins; the head's base across the direction of travel.
  const base = { x: tip.x - ux * HEAD_LENGTH, y: tip.y - uy * HEAD_LENGTH };
  const nx = -uy * (HEAD_WIDTH / 2), ny = ux * (HEAD_WIDTH / 2);
  const points = [corner ? `${start.x},${start.y} ${corner.x},${corner.y}` : `${start.x},${start.y}`, `${base.x},${base.y}`];
  return {
    shaft: points.join(" "),
    head: `${base.x + nx},${base.y + ny} ${tip.x},${tip.y} ${base.x - nx},${base.y - ny}`,
    width: SHAFT,
  };
}
