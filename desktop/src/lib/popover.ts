// Where a header button's popover goes: under the header, its right edge at the button's, and
// never past either edge of the screen. On a phone the header's buttons sit near the middle and a
// popover anchored to one ran off the left edge (Tai, 2026-10-01: "even the notification stuff
// is going to not fit at all"). The popover is `position: fixed` with this top and right.

const GUTTER = 8;

export function under(button: Element, width: number): { top: number; right: number } {
  const at = button.getBoundingClientRect();
  const bar = button.closest("header")?.getBoundingClientRect();
  const right = Math.max(GUTTER, Math.min(innerWidth - at.right, innerWidth - GUTTER - width));
  return { top: (bar?.bottom ?? at.bottom) + 6, right };
}
