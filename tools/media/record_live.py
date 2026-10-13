"""Film the real panel while it coaches a recording (tools/media/film_site.py), for the
README's live pictures: the panel runs `sidecoach panel --replay`, the installed Microsoft Edge
(Playwright's msedge channel) shows its page, and every paint is kept with when it painted,
alongside what the panel said and when (its /api/poll), for compose.py to cut and set beside
the filmed site. Never shipped; development only.

    uv run --with playwright --no-sync python tools/media/record_live.py REC CONFIG OUT \
        [--size 900x720] [--port 8812] [--actions hero]

The config is a settings file of the demo's own (a data folder of copies, never the player's);
the panel never captures the screen. ACTIONS below click or type in the page when the panel
reaches a moment (a ply, or a line the coach said).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import psutil
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]

# Name -> [(when, what)]: when is ("ply", n) once the panel shows ply n, or ("said", text) once
# the coach said a line containing the text, each with a delay in seconds; what is
# ("click", selector) or ("type", text) into the ask box and send.
ACTIONS: dict[str, list[tuple[tuple[str, object, float], tuple[str, str]]]] = {
    "hero": [],
    # Their bishop lands on g4 and the player asks about it, before the moment that turns on it
    # (5.Nxe5 two moves later), so the answer there stays theirs to find.
    "ask": [(("ply", 6, 1.0), ("type", "What is their bishop doing on g4?"))],
}


def _get(port: int, path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return json.load(r)


def _act(page: Page, what: tuple[str, str]) -> None:
    kind, arg = what
    if kind == "click":
        page.click(arg)
    elif kind == "type":
        box = page.get_by_placeholder("Ask the coach")
        box.click()
        box.type(arg, delay=45)
        box.press("Enter")


def record(
    rec: Path,
    config: Path,
    out: Path,
    size: str,
    port: int,
    actions: str,
    cap: float = 0.0,
    captions: str = "",
) -> None:
    """`cap`: stop when the session's model calls cost more than this many dollars (0: none).
    `captions`: also film the captions' page (/captions) at this size, "WxH" (the config must
    have captions on)."""
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "panel"
    frames_dir.mkdir(exist_ok=True)
    width, height = (int(v) for v in size.split("x"))
    env = {**os.environ, "SIDECOACH_CONFIG": str(config)}
    log = (out / "panel.log").open("w", encoding="utf-8")
    last_tick = (rec / "index.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    length = float(json.loads(last_tick)["ts"])
    launched = time.time()
    panel = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sidecoach",
            "panel",
            "--replay",
            str(rec),
            "--port",
            str(port),
            "--no-persist",
            "--config",
            str(config),
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    events: list[dict] = []
    frames: list[tuple[float, str]] = []
    stop = threading.Event()
    # The model calls' tokens and dollars (an AI coach in the config), and the cap on them.
    usage: dict = {}
    capped = threading.Event()

    def poll() -> None:
        since, ply = 0, None
        while not stop.is_set():
            try:
                d = _get(port, f"/api/poll?since={since}")
            except Exception:  # noqa: BLE001 - not up yet, or going down
                time.sleep(0.3)
                continue
            now = time.time()
            s = d.get("state", {})
            llm = (s.get("health") or {}).get("llm")
            if llm:
                usage.update(llm)
                if cap and (llm.get("cost") or 0) > cap:
                    events.append({"t": now, "kind": "cap", "usage": llm})
                    capped.set()
            if s.get("ply") != ply:
                ply = s.get("ply")
                events.append({"t": now, "kind": "ply", "ply": ply})
            for line in d.get("lines", []):
                since = max(since, line.get("seq", 0))
                chan, level = line.get("chan"), line.get("level")
                if chan in ("coach", "game") and level == "info":
                    events.append(
                        {
                            "t": now,
                            "kind": chan,
                            "text": line.get("text", ""),
                            "data": line.get("data", {}),
                        }
                    )
            time.sleep(0.15)

    try:
        for _ in range(120):
            try:
                _get(port, "/api/health")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        poller = threading.Thread(target=poll, daemon=True)
        poller.start()
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="msedge")

            def film(
                path: str, w: int, h: int, folder: Path, kept: list, scale: int = 1
            ) -> tuple[Page, Any]:
                """A page of its own (its own context, so it paints while another is in
                front), every paint kept in `folder` with when it painted; `scale` pixels to
                the CSS pixel (2: sharp when enlarged)."""
                pg = browser.new_context(
                    viewport={"width": w, "height": h}, device_scale_factor=scale
                ).new_page()
                if path != "/":
                    # The sidecar serves its page at the routes a browser opens; the shell's own
                    # windows (/captions) it does not, so the page's own index answers and its
                    # router shows the route, as in the shell.
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                        index = r.read()
                    pg.route(
                        f"**{path}",
                        lambda route: route.fulfill(
                            status=200, content_type="text/html", body=index
                        ),
                    )
                pg.goto(f"http://127.0.0.1:{port}{path}")
                session = pg.context.new_cdp_session(pg)

                def on_frame(event: dict) -> None:
                    name = f"{len(kept):05d}.png"
                    (folder / name).write_bytes(base64.b64decode(event["data"]))
                    kept.append((event["metadata"]["timestamp"], name))
                    session.send("Page.screencastFrameAck", {"sessionId": event["sessionId"]})

                session.on("Page.screencastFrame", on_frame)
                session.send(
                    "Page.startScreencast",
                    {"format": "png", "maxWidth": w * scale, "maxHeight": h * scale},
                )
                return pg, session

            page, cdp = film("/", width, height, frames_dir, frames)
            caption_frames: list[tuple[float, str]] = []
            if captions:
                cw, ch = (int(v) for v in captions.split("x"))
                (out / "captions").mkdir(exist_ok=True)
                cpage, ccdp = film("/captions", cw, ch, out / "captions", caption_frames, scale=2)
                # A double-click shows more lines, as the user would for the long answers.
                cpage.wait_for_timeout(1500)
                box = cpage.evaluate(
                    "() => { const els = [...document.querySelectorAll('body *')]"
                    ".filter(e => { const r = e.getBoundingClientRect();"
                    " return r.width > 300 && r.height > 40 && r.height < 400; });"
                    " const r = els[els.length - 1].getBoundingClientRect();"
                    " return [r.x + r.width / 2, r.y + r.height - 18]; }"
                )
                cpage.mouse.dblclick(box[0], box[1])
            todo = list(ACTIONS[actions])
            while time.time() - launched < 900:
                page.wait_for_timeout(100)
                for item in list(todo):
                    (kind, key, delay), what = item
                    hit = [
                        e
                        for e in events
                        if (
                            (kind == "ply" and e["kind"] == "ply" and e["ply"] == key)
                            or (kind == "said" and e["kind"] == "coach" and str(key) in e["text"])
                        )
                    ]
                    if hit and time.time() >= hit[0]["t"] + delay:
                        events.append({"t": time.time(), "kind": "action", "what": list(what)})
                        _act(page, what)
                        todo.remove(item)
                # The recording has played out (it started with the panel), and a little more;
                # or the model calls reached the cap.
                if time.time() - launched > length + 8 or capped.is_set():
                    break
            cdp.send("Page.stopScreencast")
            if captions:
                ccdp.send("Page.stopScreencast")
                (out / "captions.json").write_text(json.dumps(caption_frames), encoding="utf-8")
            browser.close()
    finally:
        stop.set()
        for child in psutil.Process(panel.pid).children(recursive=True):
            child.kill()
        panel.kill()
        log.close()
    (out / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
    (out / "events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out / "usage.json").write_text(json.dumps(usage, indent=1), encoding="utf-8")
    print(
        f"{out}: {len(frames)} frames, {len(events)} events; model calls {usage or 'none'}"
        + (" -- STOPPED AT THE CAP" if capped.is_set() else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rec", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--size", default="900x720")
    parser.add_argument("--port", type=int, default=8812)
    parser.add_argument("--actions", default="hero", choices=sorted(ACTIONS))
    parser.add_argument("--cap-usd", type=float, default=0.0)
    parser.add_argument("--captions", default="", help="film /captions too, at WxH")
    args = parser.parse_args()
    record(
        args.rec,
        args.config,
        args.out,
        args.size,
        args.port,
        args.actions,
        args.cap_usd,
        args.captions,
    )


if __name__ == "__main__":
    main()
