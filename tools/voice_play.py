"""What the player hears, with the app's own voice policy: the real `PacePlan`, fed as the panel
feeds it (`_feed_voice`: one line playing, the next made behind it), over the scenarios of
`tools/voice_sim.py`, with a voice that works as the phone's does.

The voice: one worker makes the lines in order, never more than one ahead of the line playing;
a line is made in pieces at `--pace` seconds per second of speech (the phone: 0.5 idle, 0.75-0.95
with the vision running, 1.3-2.4 with a stream playing beside it -- Tai's Pixel, 2026-10-03), and
starts once every piece will be ready before the speech ahead of it runs out (the phone's
gapless start); a line held for a fast line's end is made while it waits (`prepare`). Every line
ends in the 0.35 s pause.

The coach: `voice_sim.messages_for` -- a word on every move, a lesson on a mistake, the stuck
hints, the answer to a question -- `--late` seconds later than the vision and the model alone
make it (an engine slowed by the voice, a slow model).

    uv run --no-sync python tools/voice_play.py [--pace 0.5 1.4] [--scenario P2 G2] [--late 0]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from voice_sim import SCENARIOS, Act, Heard, Msg, messages_for, score, speech_s  # noqa: E402

from sidecoach.voice.pace import HOLD, SAY, PacePlan  # noqa: E402

TAIL_S = 0.35  # the pause after every line (KokoroTts.TAIL_SILENCE_S)
WHOLE_UNDER_S = 3.5  # a line shorter than this is one piece (the phone's `Kokoro.plan`)
PASS_S = 0.08  # a piece's fixed cost
STEP = 0.02
PAUSES: dict[tuple[int, str], float] = {}  # a line heard with a pause inside it, and how long
TRIGGER = {
    "lesson": "post_mistake",
    "praise": "post_move",
    "short": "post_move",
    "hint": "stuck",
    "answer": "user_ask",
}


def pieces(text: str) -> list[float]:
    """The speech of each piece of a line, in seconds."""
    s = speech_s(text)
    if s < WHOLE_UNDER_S:
        return [s]
    first = max(1.0, s * 0.4)
    return [first, s - first]


@dataclass
class Line:
    msg: Msg
    text: str
    prepared: bool
    made_from: float | None = None  # when its making began
    start: float | None = None  # its first sound
    end: float | None = None  # its last sound, the pause included
    gap: float = 0.0  # silence inside it: a piece made after the speech ahead ran out


class Voice:
    """The phone's KokoroTts as the plan sees it: `say`, `stop`, `drop_waiting`, `busy`,
    `waiting`, and a `prepare` for the line's word."""

    def __init__(self, pace: float, max_hold: float = float("inf")) -> None:
        self.pace = pace
        self.max_hold = max_hold
        self.lines: list[Line] = []  # given, not yet ended: the first may be playing
        self.worker_free = 0.0
        self.log: list[Heard] = []
        self.prepared: dict[str, float] = {}  # text -> when it is made
        self.started_lines: list[Line] = []

    def say(self, msg: Msg, text: str, now: float, prepared: bool) -> None:
        self.lines.append(Line(msg, text, prepared))

    def prepare(self, text: str, now: float) -> None:
        if text not in self.prepared:
            self.prepared[text] = now + self.pace * speech_s(text) + PASS_S

    def playing(self, now: float) -> Line | None:
        first = self.lines[0] if self.lines else None
        return (
            first if first is not None and first.start is not None and first.start <= now else None
        )

    def busy(self, now: float) -> bool:
        return bool(self.lines)

    def waiting(self, now: float) -> int:
        return max(0, len(self.lines) - 1) if self.lines else 0

    def stop(self, now: float, why: str) -> None:
        for line in self.lines:
            if line.start is not None and line.start <= now:
                spoken = max(0.0, now - line.start)
                total = speech_s(line.text)
                words = int(len(line.text.split()) * min(1.0, spoken / total))
                if spoken >= total:
                    self.log.append(
                        Heard(line.msg, line.text, line.start, line.start + total, words, "whole")
                    )
                else:
                    self.log.append(
                        Heard(line.msg, line.text, line.start, now, words, f"cut: {why}")
                    )
            else:
                self.log.append(
                    Heard(
                        line.msg, line.text, None, None, 0, f"silent: {why}, dropped from the voice"
                    )
                )
        self.lines = []
        self.worker_free = now

    def drop_waiting(self, now: float) -> None:
        keep = self.lines[:1]
        for line in self.lines[1:]:
            self.log.append(
                Heard(line.msg, line.text, None, None, 0, "silent: no longer the news, dropped")
            )
        self.lines = keep

    def tick(self, now: float) -> list[Line]:
        """Advance to `now`; the lines whose sound began (the panel's `on_audio_starts`)."""
        began: list[Line] = []
        # A line that has ended leaves.
        while self.lines and self.lines[0].end is not None and self.lines[0].end <= now:
            line = self.lines.pop(0)
            self.log.append(
                Heard(
                    line.msg,
                    line.text,
                    line.start,
                    line.end - TAIL_S,
                    len(line.text.split()),
                    "whole",
                )
            )
            if line.gap > 0.05:
                PAUSES[(line.msg.id, line.text)] = line.gap
        # The worker makes the next line once free, never more than one ahead of the playing.
        for i, line in enumerate(self.lines[:2]):
            if line.made_from is None and now >= self.worker_free:
                line.made_from = now
                ps = pieces(line.text)
                ready = self.prepared.pop(line.text, None)
                if line.prepared and ready is not None:
                    made = [max(now, ready)] * len(ps)
                    self.worker_free = max(now, ready)
                else:
                    t, made = now, []
                    for p in ps:
                        t += self.pace * p + PASS_S
                        made.append(t)
                    self.worker_free = t
                # The gapless start: every piece ready before the speech ahead of it runs out --
                # waited for at most `max_hold` past the first piece; past it, the line starts
                # and a later piece may come after a pause.
                gapless = made[0]
                ahead = 0.0
                for j, p in enumerate(ps):
                    gapless = max(gapless, made[j] - ahead)
                    ahead += p
                start = min(gapless, made[0] + self.max_hold)
                # A piece made after the speech ahead of it ran out plays after a pause.
                end = start
                for j, p in enumerate(ps):
                    end = max(end, made[j]) + p
                line.gap = end - start - sum(ps)
                before = self.lines[i - 1] if i > 0 else None
                if before is not None and before.end is not None:
                    start = max(start, before.end)
                line.start = start
                line.end = start + speech_s(line.text) + line.gap + TAIL_S
            if i == 1 and line.start is not None and self.lines[0].end is not None:
                if line.start < self.lines[0].end:
                    shift = self.lines[0].end - line.start
                    line.start += shift
                    line.gap = max(0.0, line.gap - shift)
                line.end = line.start + speech_s(line.text) + line.gap + TAIL_S
        for line in self.lines:
            if (
                line.start is not None
                and line.start <= now < line.start + STEP
                and line not in self.started_lines
            ):
                self.started_lines.append(line)
                began.append(line)
        return began


def run(
    acts: list[Act], pace: float, late: float, max_hold: float
) -> tuple[list[Heard], list[Msg]]:
    clock = [0.0]
    plan = PacePlan(lambda: clock[0])
    voice = Voice(pace, max_hold)
    PAUSES.clear()
    msgs = messages_for(acts, True)
    for m in msgs:
        m.ready = [t + late for t in m.ready]
    plan.new_position()

    epoch_moves = 0  # the player's moves on this board (a takeback withdraws one)
    since = 0.0
    keys: dict[int, int] = {}
    silent_log: list[Heard] = []

    def feed(now: float) -> None:
        for _ in range(2):
            busy, waiting = voice.busy(now), voice.waiting(now)
            plan.voice_state(busy, waiting)
            if busy and waiting > 0:
                return
            line = plan.next_line()
            if line is None:
                return
            msg = next(m for m in msgs if keys.get(m.id) == line.key)
            voice.say(msg, line.text, now, line.prepared)

    events: list[tuple[float, int, object]] = [(a.t, 0, a) for a in acts]
    for m in msgs:
        events += [(t, 1, (m, i)) for i, t in enumerate(m.ready)]
    events.sort(key=lambda e: (e[0], e[1]))
    end = max(e[0] for e in events) + 20
    k = 0
    now = 0.0
    while now < end:
        clock[0] = now
        while k < len(events) and events[k][0] <= now:
            _, typ, obj = events[k]
            k += 1
            if typ == 0:
                act: Act = obj  # type: ignore[assignment]
                if act.what == "move":
                    epoch_moves += 1
                    stop = plan.player_moved(2 * epoch_moves - 1, now - since)
                    plan.opponent_moved(2 * epoch_moves)  # the reply, at once in a puzzle
                elif act.what == "takeback":
                    epoch_moves -= 1
                    stop = plan.taken_back(2 * epoch_moves)
                elif act.what == "new":
                    epoch_moves = 0
                    stop = plan.new_position()
                else:
                    continue  # a question: the answer is a message
                since = now
                if stop and voice.busy(now):
                    voice.stop(now, "cut by the plan")
                    plan.stopped()
                elif plan.waiting_stale():
                    voice.drop_waiting(now)
                feed(now)
            else:
                m, i = obj  # type: ignore[misc]
                keys.setdefault(m.id, m.id)
                kind = m.kind
                # The ply the message is about: a word on a move, the position it was played
                # from; a hint or an answer, the position the player faces.
                if kind in ("hint", "answer"):
                    ply = 2 * (m.moment % 100)
                else:
                    ply = 2 * ((m.about_move - 1) % 100)
                action = plan.offer(m.id, TRIGGER[kind], ply, i, m.sentences[i], streamed=m.model)
                if action == SAY:
                    feed(now)
                elif action == HOLD:
                    voice.prepare(m.sentences[i], now)
                    silent_log.append(
                        Heard(m, m.sentences[i], None, None, 0, "held: for the line's end")
                    )
                else:
                    silent_log.append(
                        Heard(m, m.sentences[i], None, None, 0, "silent: text only (plan)")
                    )
        plan.tick()
        feed(now)
        for _ in voice.tick(now):
            feed(now)  # a line began: the next goes to the voice to be made meanwhile
        now += STEP
    voice.stop(end, "end")
    # A held word that was said appears twice: once held, once heard; keep the hearing.
    heard_ids = {(h.msg.id, h.sentence) for h in voice.log if h.start is not None}
    log = voice.log + [h for h in silent_log if (h.msg.id, h.sentence) not in heard_ids]
    return log, msgs


def show(log: list[Heard], acts: list[Act]) -> None:
    rows: list[tuple[float, str]] = []
    n = 0
    for a in acts:
        if a.what == "move":
            n += 1
        label = {
            "move": f"MOVE {n}{'' if a.good else ' (wrong)'}{' (a find)' if a.exciting else ''}",
            "takeback": "TAKEBACK",
            "new": "NEXT PUZZLE",
            "ask": "ASKS",
        }[a.what]
        rows.append((a.t, f"   {a.t:6.1f}            >> {label}"))
    for h in log:
        at = (
            h.start
            if h.start is not None
            else h.msg.ready[min(len(h.msg.ready) - 1, h.msg.sentences.index(h.sentence))]
            if h.sentence in h.msg.sentences
            else h.msg.asked_at
        )
        if h.start is not None:
            text = (
                h.sentence
                if h.why == "whole"
                else " ".join(h.sentence.split()[: h.words_heard]) + " ..."
            )
            when = f"{h.start:6.1f}-{h.end:5.1f}"
            delay = h.start - h.msg.asked_at
            pause = PAUSES.get((h.msg.id, h.sentence))
            why = f"{h.why} ({pause:.1f}s pause in it)" if pause and h.why == "whole" else h.why
            rows.append(
                (
                    at,
                    f"   {when}  heard [{h.msg.kind:<6}] +{delay:4.1f}s {why[:24]:<24} {text[:80]}",
                )
            )
        else:
            rows.append(
                (
                    at,
                    f"   {at:6.1f}       not heard [{h.msg.kind:<6}] "
                    f"{h.why[:40]:<40} {h.sentence[:60]}",
                )
            )
    for _, row in sorted(rows, key=lambda r: r[0]):
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--pace", nargs="+", type=float, default=[0.5, 1.4])
    ap.add_argument("--late", type=float, default=0.0)
    ap.add_argument(
        "--max-hold",
        type=float,
        default=float("inf"),
        help="the most the first sound waits past the first piece for the line to be gapless",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    table = []
    for key in args.scenario:
        title, acts = SCENARIOS[key]
        for pace in args.pace:
            log, msgs = run(acts, pace, args.late, args.max_hold)
            table.append((key, pace, score(log, msgs, acts)))
            if not args.quiet:
                print(f"\n=== {key} at voice pace {pace}: {title}")
                show(log, acts)
    cols = ["lesson", "praise", "hint", "answer", "cut", "stale", "spoiler", "overtalk"]
    print("\nscores (whole+partial/total; cut, stale, spoiler, overtalk: lower is better)")
    print(f"{'':4} {'pace':>5}" + "".join(f"{c:>10}" for c in cols))
    for key, pace, sc in table:
        print(f"{key:4} {pace:>5}" + "".join(f"{sc.get(c, '-'):>10}" for c in cols))


if __name__ == "__main__":
    main()
