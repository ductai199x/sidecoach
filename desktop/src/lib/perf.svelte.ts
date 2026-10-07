// The page's own meter (?perf=<label>), for the shell's evaluation (docs/app.md,
// section 4): frames per second from requestAnimationFrame, frames longer than 50 ms,
// and the poll round trip, shown in the status strip, kept on window.__perf, and posted
// every five seconds to /api/perf, which appends them to data/perf/<label>.jsonl.

import { isTauri } from "@tauri-apps/api/core";
import { panel } from "./api";
import { store } from "./panel.svelte";

class Perf {
  on = $state(false);
  text = $state("–");
  private label = "";
  private frames = 0;
  private framesTotal = 0;
  private longTotal = 0;
  private fpsMin = Infinity;
  private fpsSum = 0;
  private fpsN = 0;
  private last = 0;
  private second = 0;

  start(label: string): void {
    if (this.on) return;
    this.on = true;
    this.label = label;
    this.last = this.second = performance.now();
    const pct = (arr: number[], p: number) => { if (!arr.length) return null; const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(s.length * p))]; };
    const frame = (now: number) => {
      const dt = now - this.last; this.last = now;
      this.frames++; this.framesTotal++;
      if (dt > 50) this.longTotal++;
      if (now - this.second >= 1000) {
        const fps = (this.frames * 1000) / (now - this.second);
        this.fpsMin = Math.min(this.fpsMin, fps); this.fpsSum += fps; this.fpsN++;
        this.frames = 0; this.second = now;
        const p50 = pct(store.polls, 0.5), p95 = pct(store.polls, 0.95);
        this.text = fps.toFixed(0) + " fps (min " + this.fpsMin.toFixed(0) + "), long " + this.longTotal + ", poll " + (p50 == null ? "–" : p50.toFixed(0) + "/" + p95!.toFixed(0) + " ms");
        // What the page holds, for a soak: the JS heap (Chromium's non-standard
        // performance.memory), the DOM's size, and the store's lists.
        const mem = (performance as any).memory;
        (window as any).__perf = {
          fps, fps_min: this.fpsMin, fps_mean: this.fpsSum / this.fpsN, long_frames: this.longTotal, frames: this.framesTotal, poll_p50: p50, poll_p95: p95,
          heap_mb: mem ? Math.round(mem.usedJSHeapSize / 1048576) : null, dom: document.getElementsByTagName("*").length,
          rows: store.rows.length, coach: store.coach.length, convo: store.convo.length,
        };
      }
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    setInterval(() => {
      const sample = (window as any).__perf;
      if (!sample) return;
      const host = isTauri() ? "tauri" : /Electron/.test(navigator.userAgent) ? "electron" : "browser";
      panel.perf(this.label, { host, ...sample, ua: navigator.userAgent }).catch(() => {});
    }, 5000);
  }
}

export const perf = new Perf();
