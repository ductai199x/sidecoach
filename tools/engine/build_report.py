"""The engine redesign report, built from the measurements' own files (data/engine-cache/):
before and after on both platforms, accuracy against ground truth, the scheduling front.

    uv run --no-sync python tools/engine/build_report.py <out.html>
"""

from __future__ import annotations

import html
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
C = ROOT / "data" / "engine-cache"
SIX = 1.13


def q(v: list[float], p: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def stats(v: list[float]) -> dict[str, float]:
    return {
        "p50": q(v, 0.5),
        "p90": q(v, 0.9),
        "p99": q(v, 0.99),
        "max": max(v),
        "over5": sum(x > 5 for x in v),
        "n": len(v),
        "sum": sum(v),
    }


def desktop() -> dict[str, dict[str, float]]:
    single = json.loads((C / "single.json").read_text(encoding="utf-8"))
    keys = {r["key"] for r in single}
    db = sqlite3.connect(C / "clean.sqlite")
    six, threats, m12 = [], [], []
    for k, f in db.execute("SELECT key, found FROM clean"):
        if k not in keys:
            continue
        x = json.loads(f)
        six.append(x["6"]["wall"] / SIX)
        threats.append(x["threats"]["wall"] / SIX)
        m12.append(next((t for d, t in x["6"]["depths"] if d >= 12), x["6"]["wall"]) / SIX)
    db.close()
    return {
        "before": stats([a + b for a, b in zip(six, threats, strict=True)]),
        "one": stats([r["one"] for r in single]),
        "hint": stats(m12),
    }


def phone() -> dict[str, dict[str, float]]:
    rows = json.loads((C / "phone-tail.json").read_text(encoding="utf-8"))["rows"]
    rnd = rows[150:]
    return {
        "before": stats([r["six"]["wall"] + r["threats"]["wall"] for r in rows]),
        "one": stats([r["one"]["wall"] for r in rows]),
        "hint": stats([r["six12"]["wall"] for r in rows]),
        "before_rnd": stats([r["six"]["wall"] + r["threats"]["wall"] for r in rnd]),
        "one_rnd": stats([r["one"]["wall"] for r in rnd]),
        "hint_rnd": stats([r["six12"]["wall"] for r in rnd]),
        "cpu_before": sum(r["six"]["cpu"] for r in rows),
        "cpu_one": sum(r["one"]["cpu"] for r in rows),
        "one18_rnd": stats(
            [
                r["one"]["wall"]
                for r in json.loads((C / "phone-one-18.json").read_text(encoding="utf-8"))["rows"]
            ]
        ),
    }


def front_rows() -> str:
    """The scheduling policies worth a look, per kind of workload, from sweep2's summary."""
    path = C / "sweep2_summary.json"
    if not path.exists():
        return "<tr><td>still running</td></tr>"
    rows = {r["name"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    # Only policies without deep candidates: with them, up to 1,163 candidate positions a session
    # were never measured and were left out, which flatters their latency -- not shown.
    picks = [
        ("3 workers, on demand (no look-ahead)", "w3 r0 c0 look0 reach0.15 newest"),
        ("3 workers, look-ahead 1 ply (reach 0.3)", "w3 r1 c0 look1 reach0.3 verdicts"),
        ("4 workers, look-ahead 2 plies (reach 0.15)", "w4 r1 c0 look2 reach0.15 newest"),
    ]
    out = [
        "<tr><th>policy / workload</th><th>cores</th><th>verdict p90</th><th>hint p90</th>"
        "<th>deep p90</th><th>wasted</th></tr>"
    ]
    for label, name in picks:
        r = rows.get(name)
        if r is None:
            continue
        out.append(f'<tr class="group"><th colspan=6>{html.escape(label)}</th></tr>')
        for kind in (
            "puzzles",
            "600",
            "900+10",
            "180+2",
            "blitz x3",
            "burst 0.5s",
            "burst 0.2s",
            "browsing",
        ):
            a = r["kinds"].get(kind)
            if a is None:
                continue

            def s(x: float | None) -> str:
                return "–" if x is None else f"{x:.2f} s"

            out.append(
                f"<tr><th>&nbsp;&nbsp;{kind}</th><td>{a['cores']:.2f}</td><td>{s(a['verdict_user']['p90'])}</td>"
                f"<td>{s(a['hint']['p90'])}</td><td>{s(a['deep']['p90'])}</td><td>{a['wasted']:.0%}</td></tr>"
            )
    return "".join(out)


def games() -> list[tuple[int, float]]:
    from sidecoach.analysis.score import Thresholds, severity

    rows = json.loads((C / "quality_games.json").read_text(encoding="utf-8"))
    th = Thresholds()

    def loss(r: dict, d: int) -> float | None:
        b = r["before"].get(str(d))
        if not r["after"]:
            return 0.0 if b is not None else None
        a = r["after"].get(str(d))
        return None if b is None or a is None else max(0.0, b - (1.0 - a))

    out = []
    for d in range(10, 24):
        same = n = 0
        for r in rows:
            t, x = loss(r, 24), loss(r, d)
            if t is None or x is None:
                continue
            n += 1
            same += severity(t, th) == severity(x, th)
        out.append((d, same / n))
    return out


def puzzles() -> tuple[int, list[tuple[int, int]]]:
    path = C / "quality_puzzles.json"
    if not path.exists():
        return 0, []
    out = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for d in range(10, 19):
        solved = sum(1 for r in out if all(s["right"].get(str(d), False) for s in r["steps"]))
        rows.append((d, solved))
    return len(out), rows


def coach_rows() -> str:
    """The AI coach's requests by workload (requests.txt: four workers, the coach on)."""
    import re

    path = C / "requests.txt"
    if not path.exists():
        return "<tr><td>not measured</td></tr>"
    text = path.read_text(encoding="utf-8").split("== workers 4", 1)[-1]
    rows: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        m = re.match(
            r"(\S+(?: \S+)?)\s+(lesson|question|their_turn|verdict)\s+n\s+\d+\s+p50 ([\d.]+) p90 ([\d.]+) max\s+([\d.]+)",
            line,
        )
        if m:
            rows.setdefault(m.group(1), {})[m.group(2)] = (
                f'{m.group(4)} s <span class="lede">(max {m.group(5)})</span>'
            )
    out = [
        "<tr><th>workload</th><th>verdict p90</th><th>lesson prompt p90</th><th>question, their turn p90</th><th>move asked about p90</th></tr>"
    ]
    for kind, r in rows.items():
        out.append(
            f"<tr><th>{html.escape(kind)}</th><td>{r.get('verdict', '–')}</td><td>{r.get('lesson', '–')}</td>"
            f"<td>{r.get('their_turn', '–')}</td><td>{r.get('question', '–')}</td></tr>"
        )
    return "".join(out)


def glue_rows(main: str, branch: str, depth: int) -> str:
    """The built apps end to end (bench_glue.py, GlueBench): each answer's wait from the move,
    the depth it reached, the engines' CPU -- main (as installed) against this branch."""

    def load(name: str) -> list[dict]:
        data = json.loads((C / name).read_text(encoding="utf-8"))
        return data["rows"] if isinstance(data, dict) else data

    out = [
        "<tr><th></th><th>verdict p50 / p90 / worst</th><th>first answer p50 / p90 / worst</th>"
        "<th>whole p50 / p90 / worst</th><th>CPU, 120 moves</th><th>below depth</th></tr>"
    ]
    for label, name, strong in (("Before (main, 0.1.76)", main, False), ("After", branch, True)):
        rows = load(name)

        def cell(key: str, rows: list[dict] = rows) -> str:
            v = [r[key] for r in rows if key in r]
            return f"{q(v, 0.5):.2f} / {q(v, 0.9):.2f} / {max(v):.2f} s"

        below = sum(r.get("verdict_depth", depth) < depth for r in rows) + sum(
            r.get("whole_depth", depth) < depth for r in rows
        )
        low = min([r[k] for r in rows for k in ("verdict_depth", "whole_depth") if k in r])
        cls = ' class="after"' if strong else ""
        out.append(
            f"<tr{cls}><th>{label}</th><td>{cell('verdict')}</td><td>{cell('first')}</td>"
            f"<td>{cell('whole')}</td><td>{sum(r['cpu'] for r in rows):.0f} s</td>"
            f"<td>{below} answers{f' (as low as {low})' if below else ''}</td></tr>"
        )
    return "".join(out)


def fmt(x: float) -> str:
    return f"{x:.2f}"


def row(name: str, s: dict[str, float], strong: bool = False) -> str:
    cls = ' class="after"' if strong else ""
    return (
        f"<tr{cls}><th>{html.escape(name)}</th><td>{fmt(s['p50'])} s</td><td>{fmt(s['p90'])} s</td>"
        f"<td>{fmt(s['p99'])} s</td><td>{s['max']:.1f} s</td><td>{s['over5']} of {s['n']}</td></tr>"
    )


def build(out: Path) -> None:
    d, p, g = desktop(), phone(), games()
    n_puz, puz = puzzles()
    head = "<tr><th></th><th>p50</th><th>p90</th><th>p99</th><th>worst</th><th>over 5 s</th></tr>"
    puz_rows = "".join(
        f"<tr><th>depth {dd}</th><td>{s}/{n_puz}</td><td>{s / n_puz:.3%}</td><td>{n_puz - s}</td></tr>"
        for dd, s in puz
    )
    game_rows = "".join(f"<tr><th>depth {dd}</th><td>{a:.1%}</td></tr>" for dd, a in g)
    page = TEMPLATE.format(
        desk_rows=head
        + row("Before: six lines, depth 18, + threats", d["before"])
        + row("After: one line, depth 18 (verdict, best move)", d["one"], True)
        + row("After: six lines, depth 12 (the hint)", d["hint"], True),
        phone_rows=head
        + row("Before: six lines, depth 14, + threats", p["before"])
        + row("After: one line, depth 14", p["one"], True)
        + row("After: six lines, depth 12 (the hint)", p["hint"], True),
        phone_rnd=head
        + row("Before", p["before_rnd"])
        + row("After: one line", p["one_rnd"], True)
        + row("After: the hint", p["hint_rnd"], True),
        phone_cpu=f"{p['cpu_before']:.0f} s → {p['cpu_one']:.0f} s",
        phone18=head
        + row("Before: six lines, depth 14, + threats", p["before_rnd"])
        + row("After: one line, depth 14", p["one_rnd"], True)
        + row("After: one line, depth 18", p["one18_rnd"], True),
        front=front_rows(),
        coach=coach_rows(),
        desk_cpu=f"{d['before']['sum']:.0f} s → {d['one']['sum']:.0f} s",
        n_desk=d["one"]["n"],
        puz_rows=puz_rows or "<tr><td colspan=4>still running</td></tr>",
        n_puz=n_puz,
        game_rows=game_rows,
        glue_desk=glue_rows("bench_glue-main.json", "bench_glue-branch.json", 18),
        glue_phone=glue_rows("phone-glue-main.json", "phone-glue-branch.json", 14),
    )
    out.write_text(page, encoding="utf-8")


TEMPLATE = """<title>Engine Redesign</title>
<style>
:root {{ --bg:#f6f7f4; --ink:#1d2321; --soft:#5d6863; --line:#d9ddd6; --accent:#1f6f54; --after:#e5f1ea; --warn:#9a4b16; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121614; --ink:#e7ece9; --soft:#9aa5a0; --line:#2b332f; --accent:#6fcf9f; --after:#173226; --warn:#f0a868; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#121614; --ink:#e7ece9; --soft:#9aa5a0; --line:#2b332f; --accent:#6fcf9f; --after:#173226; --warn:#f0a868; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.55 "IBM Plex Sans",system-ui,sans-serif; margin:0; padding:24px 16px 64px; }}
main {{ max-width:880px; margin:0 auto; }}
h1 {{ font-size:28px; margin:0 0 4px; text-wrap:balance; }}
h2 {{ font-size:19px; margin:36px 0 8px; border-top:1px solid var(--line); padding-top:20px; }}
p, li {{ max-width:68ch; }}
.lede {{ color:var(--soft); }}
.wrap {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; margin:8px 0; }}
th, td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child {{ text-align:left; font-weight:500; white-space:normal; }}
tr.after {{ background:var(--after); }}
.key {{ color:var(--accent); font-weight:600; }}
.warn {{ color:var(--warn); }}
code {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:13px; }}
</style>
<main>
<h1>Engine Redesign</h1>
<p class="lede">Stockfish work in Sidecoach, from first principles: what made it slow, what it costs before and after on the desktop and the phone, and how accurate each choice is against ground truth. Every number below is measured on real Stockfish, never assumed.</p>

<h2>The root cause</h2>
<p>Every position was searched with <b>six lines to depth 18</b>. Asking Stockfish for the exact depth-18 value of several moves at once makes the search <b>explode</b> on decided or forcing positions: one in-check position took 53 s on two lines, against 0.18 s on one line. Over {n_desk} positions from real sessions, 84 six-line searches took over 5 s (worst 47 s). One line to depth 18 does not explode.</p>
<p>Two more findings: extra threads never pay at a fixed depth (one thread is fastest per answer and per watt), and the silence watchdog killed searches that were still working (one root move searched 48 s without a word) — now fixed on both platforms.</p>

<h2>The built apps, end to end</h2>
<p>The app's own engine glue, with its default settings, fed three of your 10-minute games (40 plies each), every move from idle: how long after the move its verdict, the first word on the new position and the position whole came, the depth each answer really reached, and Stockfish's CPU. The same benchmark ran against <code>main</code> (0.1.76, as installed) and the rebuilt engine work. <code>main</code> caps every search's time (5 seconds on the desktop) and cuts the depth short to keep to it; the rebuild never cuts a depth.</p>
<p><b>Desktop</b> (depth 18; before: one engine of four threads; after: three single-thread workers):</p>
<div class="wrap"><table>{glue_desk}</table></div>
<p><b>Phone</b> (Pixel 10 Pro XL, depth 14; before: one engine of two threads; after: two single-thread workers):</p>
<div class="wrap"><table>{glue_phone}</table></div>

<h2>Desktop: before and after <span class="lede">(cold, one thread, {n_desk} positions)</span></h2>
<div class="wrap"><table>{desk_rows}</table></div>
<p>CPU for those searches: <span class="key">{desk_cpu}</span>.</p>

<h2>Phone: before and after <span class="lede">(Pixel 10 Pro XL, real, 300 positions incl. the 150 hardest)</span></h2>
<div class="wrap"><table>{phone_rows}</table></div>
<p>The 150 random positions alone, typical play:</p>
<div class="wrap"><table>{phone_rnd}</table></div>
<p>CPU for those searches: <span class="key">{phone_cpu}</span>.</p>

<p>The phone can now afford depth 18 for its verdicts — one line is cheaper than its old six at depth 14 — on the same 150 random positions:</p>
<div class="wrap"><table>{phone18}</table></div>

<h2>Accuracy: puzzles</h2>
<p>{n_puz} Lichess puzzles; solved at a depth when every solver move is the move that depth chose (any mate on the last move).</p>
<p>Depth 18 solves 99.83%. Of its 34 misses, a depth-24 check found 2 where the engine's move also wins; 32 are real misses (rated 1,800–2,900), where the engine plays a drawing move and the puzzle wins. 99.83% is accepted (your call): puzzles are measured at depth 18 like every other position.</p>
<div class="wrap"><table><tr><th></th><th>solved</th><th>rate</th><th>failed</th></tr>{puz_rows}</table></div>

<h2>Accuracy: game verdicts</h2>
<p>709 moves of 12 of your games; each depth's severity against the severity at depth 24. Even depth 23 disagrees with 24 on 2.4%: that is the engine's own noise at the thresholds, not error.</p>
<div class="wrap"><table><tr><th></th><th>same severity as depth 24</th></tr>{game_rows}</table></div>

<h2>Scheduling: the front <span class="lede">(desktop, simulated on the measured searches)</span></h2>
<p>Every position: one line to depth 18. The position you face: also six lines to depth 12 (the hint at once), its threats, and — with "deep candidates" — each of its six best moves' own one-line search (the coach's candidates at full depth, and exactly the positions the game goes to next). The virtual clock was checked by hand on five worked games. Latency is from the moment an answer is needed; cores are the average Stockfish load over the session.</p>
<div class="wrap"><table>{front}</table></div>
<p>With searches this cheap, look-ahead buys a few hundredths of a second and throws away a third of its work: computing on demand is the front.</p>
<p><b>Deep candidates</b> (each of the six best moves' own one-line search, for the coach's lessons and the review) are not in this table: the simulation lacked measurements for too many of those positions to be trusted. Measured directly instead, on 1,000 real positions, three workers side by side: <span class="key">p50 0.42 s, p90 0.77 s, p99 1.86 s</span>, worst 10.6 s (a decided position); 6% less CPU than one six-line search, which took p50 1.12 s, p99 5.8 s, worst 47 s on the same positions.</p>

<h2>The AI coach's own requests <span class="lede">(desktop, simulated)</span></h2>
<p>What the model's prompt needs, on both turns, through the same pool and never a second Stockfish: after each of your moves, the lesson's prompt (the position you moved from, whole; the position after, its best line and threats); on the opponent's turn, a question asked the moment it begins (their position, whole); and a move you or the model ask about (that move's own one-line search). Verdicts go first; a request about a moment the game has moved past is let go — before that rule, bursts piled requests up to 14 s and pushed verdicts back with them.</p>
<div class="wrap"><table>{coach}</table></div>

<h2>What was built <span class="lede">(both platforms, branch <code>stockfish-restructure</code>)</span></h2>
<ul>
<li><b>Every search is one line</b> to the full depth: the best line (the verdict, the best move); the hint, six lines to depth 12 at once; each candidate move searched as its own position, one line, in parallel — never a multi-line search at the full depth. A position with one legal move is read from its child, never searched.</li>
<li><b>One order of work</b>: the next verdict's searches first, then the coach's questions, then the position on the board, then the rest of the game (whose review then searches nothing). Each position is told at once, then whole, then with its threats.</li>
<li><b>Decided positions</b>: a best-line search still running after 1 s whose deepest depth shows the position decided judges the move with that score (your call; no larger error than depth 18's own noise).</li>
<li><b>The AI coach</b> asks the same pool for every position it or you ask about; its second Stockfish (three lines, which explode as six do) is gone. The offline coach, game review, retries, puzzles and the profile read the same searches.</li>
<li><b>Workers</b>: 3 single-thread on the desktop — they answer as fast as four or six (<code>tools/engine/workers_compare.py</code>; four only finish a whole position 0.1 s sooner) — and 2 on the phone. No look-ahead beyond the candidates. The phone's depth stays a setting (14 by default).</li>
<li><b>Game review</b>: each engine finishes one position at a time, so a review stopped part way keeps every position it finished.</li>
<li><b>Fixed along the way</b>: the watchdog (silence <i>and</i> idle CPU), a guard against Git Bash path mangling, a retried move searching positions its review had measured, the hint kept when the settings changed under it.</li>
</ul>
</main>
"""


if __name__ == "__main__":
    build(Path(sys.argv[1]))
