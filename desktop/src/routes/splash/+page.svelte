<script lang="ts">
  // The splash screen: a frameless card over the desktop while the sidecar starts and
  // warms up (the models, Stockfish, the coach), so the main window opens ready. The
  // shell opens this window before the sidecar is spawned and hides the main window
  // until `splash_done` (desktop/src-tauri/src/lib.rs). The progress is a rank of the
  // board: eight squares, one per warm-up stage, lit as each finishes.
  import { onDestroy, onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";

  type Stage = { name: string; label: string; state: "pending" | "loading" | "done" | "off"; detail?: string; seconds?: number };
  type Warm = { done: boolean; seconds: number; stages: Stage[] };
  type Status = { port: number | null; error: string | null };

  const PLACEHOLDER: Stage[] = [
    "Piece recognition", "Voice", "Ear", "Microphone", "Board reading", "Stockfish", "Coach", "Ready",
  ].map((label) => ({ name: label, label, state: "pending" }));

  let port = $state<number | null>(null);
  let error = $state("");
  let warm = $state<Warm | null>(null);
  let started = $state(performance.now());
  let seconds = $state(0);
  let finishing = $state(false);
  let timer: ReturnType<typeof setInterval> | null = null;

  const stages = $derived(warm?.stages?.length ? warm.stages : PLACEHOLDER);
  // White's first rank, a1 to h1: each stage that finishes puts its piece on its square.
  const RANK = ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"];
  const current = $derived(stages.find((s) => s.state === "loading") ?? null);
  const line = $derived.by(() => {
    if (error) return error;
    if (!port) return "Starting…";
    if (warm?.done) {
      const off = stages.filter((s) => s.state === "off" && s.name !== "ready");
      return off.length ? `Ready, without ${off.map((s) => s.label.toLowerCase()).join(", ")}` : "Ready";
    }
    if (current) return `${current.label}…`;
    return "Warming up…";
  });

  // `?demo` (a browser, no shell): the stages advance on a timer, one every 900 ms, and
  // the card stays -- to look at the design without launching the app.
  const demo = typeof location !== "undefined" && location.search.includes("demo");
  let demoStep = 0;

  async function tick() {
    seconds = (performance.now() - started) / 1000;
    if (demo) {
      port = 1;
      const n = Math.min(PLACEHOLDER.length, Math.floor(seconds / 0.9));
      warm = {
        done: false,
        seconds,
        stages: PLACEHOLDER.map((s, i) => ({ ...s, name: s.label, state: i < n ? "done" : i === n ? "loading" : "pending" })),
      };
      demoStep = n;
      return;
    }
    if (!port) {
      const status = await invoke<Status>("sidecar_status").catch(() => null);
      if (status?.error) error = status.error;
      if (status?.port) port = status.port;
      return;
    }
    try {
      const res = await fetch(`http://127.0.0.1:${port}/api/health`);
      const health = (await res.json()) as { warm?: Warm };
      if (health.warm) warm = health.warm;
    } catch {
      /* the next tick will find it */
    }
    if (warm?.done && !finishing) {
      finishing = true;
      setTimeout(done, 700);
    }
  }

  function done() {
    if (timer) clearInterval(timer);
    timer = null;
    void invoke("splash_done").catch(() => {});
  }

  function onKey(e: KeyboardEvent) {
    if (e.key === "Escape") done();
  }

  // The background: ribbons of light -- four slow waves flowing across the card with an
  // additive glow, a voice visualiser at rest -- and a pulse that ripples out across them
  // each time a stage finishes and its piece lands on the rank. Drawn to a buffer first,
  // then put on the card in one go.
  let canvas = $state<HTMLCanvasElement | null>(null);
  let raf = 0;
  let pulses: number[] = []; // when each pulse began
  let landed = 0;

  function waves(canvasEl: HTMLCanvasElement): () => void {
    const got = canvasEl.getContext("2d");
    if (!got) return () => {};
    const W = canvasEl.width, H = canvasEl.height;
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    type Ribbon = { rgb: string; base: number; amp: number; speed: number; k1: number; k2: number; phase: number; width: number };
    const ribbons: Ribbon[] = [
      { rgb: "242, 180, 65", base: 0.8, amp: 28, speed: 0.4, k1: 1.4, k2: 2.9, phase: 0.0, width: 3 },
      { rgb: "129, 182, 76", base: 0.84, amp: 23, speed: 0.32, k1: 1.1, k2: 3.4, phase: 2.1, width: 2.5 },
      { rgb: "91, 139, 176", base: 0.76, amp: 33, speed: 0.27, k1: 0.9, k2: 2.2, phase: 4.0, width: 2.5 },
      { rgb: "247, 199, 94", base: 0.88, amp: 18, speed: 0.5, k1: 1.8, k2: 4.1, phase: 1.3, width: 1.5 },
    ];
    const t0 = performance.now();
    const buf = document.createElement("canvas");
    buf.width = W; buf.height = H;
    const bctx = buf.getContext("2d") as CanvasRenderingContext2D;

    // The chess in it: pieces made of light. Some two thousand points form a piece --
    // a lathe profile (radius by height) turned around its axis, with the notches a rook's
    // battlements and a queen's crown need -- turn slowly for a while, blow apart, each
    // point on its own path, and gather again, slowly, as another piece. About two pieces
    // in the ten seconds the warm-up takes.
    type Profile = { rows: [number, number][]; height: number; notches?: { from: number; to: number; count: number } };
    const SHAPES: Record<string, Profile> = {
      king: { height: 1.0, rows: [
        [0.0, 0.23], [0.05, 0.23], [0.07, 0.19], [0.11, 0.17], [0.13, 0.12], [0.3, 0.095],
        [0.5, 0.08], [0.56, 0.085], [0.58, 0.13], [0.62, 0.13], [0.64, 0.1], [0.7, 0.14],
        [0.76, 0.15], [0.82, 0.12], [0.85, 0.07], [0.87, 0.05], [0.88, 0.02], [1.0, 0.02],
      ] },
      queen: { height: 0.95, notches: { from: 0.86, to: 0.94, count: 8 }, rows: [
        [0.0, 0.23], [0.05, 0.23], [0.07, 0.19], [0.11, 0.17], [0.13, 0.12], [0.3, 0.09],
        [0.5, 0.075], [0.58, 0.08], [0.6, 0.12], [0.64, 0.12], [0.66, 0.09], [0.76, 0.12],
        [0.86, 0.165], [0.94, 0.15], [0.945, 0.04], [0.96, 0.045], [1.0, 0.03],
      ] },
      rook: { height: 0.8, notches: { from: 0.8, to: 1.0, count: 6 }, rows: [
        [0.0, 0.23], [0.06, 0.23], [0.09, 0.18], [0.14, 0.15], [0.16, 0.13], [0.68, 0.12],
        [0.72, 0.15], [0.78, 0.16], [0.8, 0.155], [1.0, 0.15],
      ] },
      bishop: { height: 0.88, rows: [
        [0.0, 0.22], [0.05, 0.22], [0.07, 0.18], [0.11, 0.15], [0.13, 0.1], [0.4, 0.075],
        [0.52, 0.07], [0.55, 0.11], [0.59, 0.11], [0.61, 0.08], [0.72, 0.125], [0.84, 0.09],
        [0.9, 0.04], [0.92, 0.035], [0.96, 0.035], [1.0, 0.01],
      ] },
      pawn: { height: 0.66, rows: [
        [0.0, 0.22], [0.06, 0.22], [0.09, 0.17], [0.14, 0.13], [0.16, 0.1], [0.5, 0.08],
        [0.55, 0.12], [0.6, 0.12], [0.62, 0.08], [0.7, 0.13], [0.82, 0.145], [0.94, 0.1], [1.0, 0.02],
      ] },
    };
    const ORDER = ["king", "queen", "rook", "bishop", "pawn"];
    const N = 2200;
    type Pt = { x: number; y: number; z: number };
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const radiusAt = (rows: [number, number][], y: number): number => {
      for (let i = 1; i < rows.length; i++) {
        const [y0, r0] = rows[i - 1], [y1, r1] = rows[i];
        if (y <= y1) return r0 + ((r1 - r0) * (y - y0)) / (y1 - y0);
      }
      return rows[rows.length - 1][1];
    };
    function sample(shape: Profile): Pt[] {
      const pts: Pt[] = [];
      while (pts.length < N) {
        const y = rnd();
        const a = rnd() * Math.PI * 2;
        if (shape.notches && y > shape.notches.from && y < shape.notches.to) {
          const sector = (a / (Math.PI * 2)) * shape.notches.count;
          if (sector - Math.floor(sector) > 0.5) continue; // the gap between two merlons
        }
        const r = radiusAt(shape.rows, y) * (0.97 + rnd() * 0.06);
        pts.push({ x: r * Math.cos(a), y: y * shape.height, z: r * Math.sin(a) });
      }
      return pts;
    }
    const targets = ORDER.map((name) => sample(SHAPES[name]));
    const twinkle = Array.from({ length: N }, () => rnd() * Math.PI * 2);
    const pos: Pt[] = targets[0].map((p) => ({ ...p }));
    const from: Pt[] = targets[0].map((p) => ({ ...p }));
    const vel: Pt[] = targets[0].map(() => ({ x: 0, y: 0, z: 0 }));
    const FORM_S = 2.2, HOLD_S = 1.8, SCATTER_S = 0.9; // about five seconds a piece: two in a warm-up
    const cycle = { shape: 0, phase: "hold" as "form" | "hold" | "scatter", since: 0 };
    const PIECE = { cx: W * 0.76, base: H * 0.9, height: 390 };
    const ease = (p: number) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);

    function pieces(ctx: CanvasRenderingContext2D, t: number) {
      const age = t - cycle.since;
      let alpha = 1;
      if (cycle.phase === "hold") {
        if (age > HOLD_S) {
          // Blow apart: every point its own direction and speed.
          for (let i = 0; i < N; i++) {
            const u = rnd() * 2 - 1, phi = rnd() * Math.PI * 2, sp = 0.9 + rnd() * 1.2;
            const r = Math.sqrt(1 - u * u);
            vel[i] = { x: r * Math.cos(phi) * sp, y: u * sp, z: r * Math.sin(phi) * sp };
            from[i] = { ...pos[i] };
          }
          cycle.phase = "scatter";
          cycle.since = t;
        }
      } else if (cycle.phase === "scatter") {
        const k = Math.min(1, age / SCATTER_S);
        alpha = 1 - 0.5 * k;
        for (let i = 0; i < N; i++) {
          const d = age * (1 - 0.25 * k); // easing off a little as they fly
          pos[i] = { x: from[i].x + vel[i].x * d, y: from[i].y + vel[i].y * d, z: from[i].z + vel[i].z * d };
        }
        if (age > SCATTER_S) {
          for (let i = 0; i < N; i++) from[i] = { ...pos[i] };
          cycle.shape = (cycle.shape + 1) % ORDER.length;
          cycle.phase = "form";
          cycle.since = t;
        }
      } else {
        const k = ease(Math.min(1, age / FORM_S));
        alpha = 0.4 + 0.6 * k;
        const target = targets[cycle.shape];
        for (let i = 0; i < N; i++) {
          pos[i] = {
            x: from[i].x + (target[i].x - from[i].x) * k,
            y: from[i].y + (target[i].y - from[i].y) * k,
            z: from[i].z + (target[i].z - from[i].z) * k,
          };
        }
        if (age > FORM_S) {
          cycle.phase = "hold";
          cycle.since = t;
        }
      }
      const theta = t * 0.3, tilt = 0.2;
      const cosT = Math.cos(theta), sinT = Math.sin(theta);
      const cosX = Math.cos(tilt), sinX = Math.sin(tilt);
      const f = 2.6;
      const glow = ctx.createRadialGradient(PIECE.cx, PIECE.base - PIECE.height * 0.45, 10, PIECE.cx, PIECE.base - PIECE.height * 0.45, PIECE.height * 0.7);
      glow.addColorStop(0, `rgba(242, 180, 65, ${(0.1 * alpha).toFixed(3)})`);
      glow.addColorStop(1, "rgba(242, 180, 65, 0)");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, W, H);
      for (let i = 0; i < N; i++) {
        const p = pos[i];
        // Turn around the axis, then tip the axis a little towards the viewer.
        const x1 = p.x * cosT - p.z * sinT, z1 = p.x * sinT + p.z * cosT;
        const y1 = p.y - 0.5;
        const y2 = y1 * cosX - z1 * sinX, z2 = y1 * sinX + z1 * cosX;
        const scale = f / (f + z2);
        const sx = PIECE.cx + x1 * PIECE.height * scale;
        const sy = PIECE.base - PIECE.height * 0.5 - y2 * PIECE.height * scale;
        const depth = Math.min(1, Math.max(0, (z2 + 0.25) / 0.5)); // 0 near .. 1 far
        const tw = 0.8 + 0.2 * Math.sin(t * 1.7 + twinkle[i]);
        const a = (0.75 - 0.55 * depth) * tw * alpha;
        ctx.fillStyle = `rgba(247, 199, 94, ${a.toFixed(3)})`;
        const size = 1.3 + 1.6 * (1 - depth);
        ctx.fillRect(sx - size / 2, sy - size / 2, size, size);
      }
    }

    function y(r: Ribbon, x: number, t: number, spread: number): number {
      const u = x / W;
      return (
        H * r.base +
        r.amp * spread * (Math.sin(u * Math.PI * 2 * r.k1 + t * r.speed + r.phase) * 0.7 +
          Math.sin(u * Math.PI * 2 * r.k2 - t * r.speed * 1.3 + r.phase * 1.7) * 0.3)
      );
    }

    function frame(now: number) {
      const t = reduced ? 0 : (now - t0) / 1000;
      const ctx = bctx;
      ctx.clearRect(0, 0, W, H);
      ctx.globalCompositeOperation = "lighter";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      // The pulses: a ring of light spreading from the rank; the ribbons swell as it passes.
      const live = pulses.filter((p) => now - p < 1400);
      pulses = live;
      for (const r of ribbons) {
        // Three strokes per ribbon: a wide faint one, a mid one and the bright core.
        for (const [mult, alpha] of [[9, 0.03], [4, 0.08], [1, 0.5]] as [number, number][]) {
          ctx.strokeStyle = `rgba(${r.rgb}, ${alpha})`;
          ctx.lineWidth = r.width * mult;
          ctx.beginPath();
          for (let x = -10; x <= W + 10; x += 6) {
            let spread = 1;
            for (const p of live) {
              const age = (now - p) / 1400;
              const radius = age * W * 0.75;
              const d = Math.abs(Math.abs(x - W / 2) - radius);
              spread += Math.max(0, 1 - d / 90) * (1 - age) * 1.4;
            }
            const yy = y(r, x, t, spread);
            if (x === -10) ctx.moveTo(x, yy);
            else ctx.lineTo(x, yy);
          }
          ctx.stroke();
        }
      }
      pieces(ctx, t);
      // The pulse's own glow: a soft ring.
      for (const p of live) {
        const age = (now - p) / 1400;
        const radius = age * W * 0.75;
        const g = ctx.createRadialGradient(W / 2, H * 0.62, Math.max(0, radius - 60), W / 2, H * 0.62, radius + 40);
        g.addColorStop(0, "rgba(242, 180, 65, 0)");
        g.addColorStop(0.6, `rgba(242, 180, 65, ${(0.22 * (1 - age)).toFixed(3)})`);
        g.addColorStop(1, "rgba(242, 180, 65, 0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, W, H);
      }
      ctx.globalCompositeOperation = "source-over";
      const out = got as CanvasRenderingContext2D;
      out.clearRect(0, 0, W, H);
      out.drawImage(buf, 0, 0);
      if (!reduced || live.length) raf = requestAnimationFrame(frame);
      else raf = 0;
    }
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }

  // A pulse for every piece that lands.
  $effect(() => {
    const n = stages.filter((s) => s.state === "done").length;
    if (n > landed) {
      pulses.push(performance.now());
      if (!raf && canvas) waves(canvas);
    }
    landed = n;
  });

  onMount(() => {
    timer = setInterval(() => void tick(), 150);
    void tick();
    const stop = canvas ? waves(canvas) : () => {};
    return stop;
  });
  onDestroy(() => {
    if (timer) clearInterval(timer);
  });
</script>

<svelte:window onkeydown={onKey} />

<div class="card" class:ready={warm?.done} class:failed={!!error}>
  <div class="scene" aria-hidden="true">
    <canvas bind:this={canvas} width="720" height="480"></canvas>
  </div>
  <div class="veil" aria-hidden="true"></div>
  <img class="logo" src="/logo.svg" alt="" draggable="false" />
  <h1>Sidecoach</h1>

  <div class="rank" role="progressbar" aria-valuemin="0" aria-valuemax={stages.length} aria-valuenow={stages.filter((s) => s.state === "done").length}>
    {#each stages as s, i (s.name)}
      <div class="sq {s.state}" class:light={i % 2 === 1} title={s.detail ? `${s.label}: ${s.detail}` : s.label}>
        {#if s.state === "done" || s.state === "off"}
          <img src={`/pieces/${RANK[i] ?? "wP"}.svg`} alt={s.label} draggable="false" />
        {/if}
      </div>
    {/each}
  </div>

  <p class="line">{line}<span class="secs">{seconds.toFixed(1)} s</span></p>

  {#if error}
    <button type="button" onclick={done}>Open anyway</button>
  {/if}
</div>

<style>
  :global(html), :global(body) { background: #1c1a18 !important; margin: 0; overflow: hidden; }
  .card {
    box-sizing: border-box;
    width: 720px; height: 480px;
    margin: 0; /* the window is the card; Windows rounds and shadows it */
    padding: 44px 52px 30px;
    display: flex; flex-direction: column; align-items: center;
    background: #1c1a18;
    border-radius: 0;
    color: #e6e5e3;
    font-family: "Segoe UI", system-ui, sans-serif;
    user-select: none;
    -webkit-user-select: none;
    position: relative;
    overflow: hidden;
  }
  .card > :not(.scene):not(.veil) { position: relative; z-index: 2; }
  /* The ribbons of light behind everything, edge to edge. */
  .scene { position: absolute; inset: 0; z-index: 0; overflow: hidden; background: radial-gradient(ellipse 90% 80% at 50% 60%, #221f1b 0%, #1c1a18 100%); }
  .scene canvas { position: absolute; left: 0; top: 0; width: 720px; height: 480px; }
  /* The veil: a little shade behind the text, the edges left alone. */
  .veil {
    position: absolute; inset: 0; z-index: 1;
    background: radial-gradient(ellipse 50% 55% at 50% 40%, rgba(28, 26, 24, 0.5) 0%, rgba(28, 26, 24, 0.2) 70%, rgba(28, 26, 24, 0) 100%);
  }
  .logo {
    width: 124px; height: 124px;
    filter: drop-shadow(0 0 28px rgba(242, 180, 65, 0.35)) drop-shadow(0 6px 14px rgba(0, 0, 0, 0.5));
    animation: breathe 2.4s ease-in-out infinite;
  }
  .ready .logo { animation: none; transform: scale(1.04); transition: transform 300ms ease-out; }
  @keyframes breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.04); } }
  h1 { margin: 20px 0 0; font-size: 33px; font-weight: 650; letter-spacing: 0.01em; }

  /* The first rank: a1 dark, then alternating, in the board's own colours dimmed until
     the stage is done; the piece lands when it is. */
  .rank { display: flex; gap: 0; margin-top: 38px; border: 2px solid #2a2724; border-radius: 4px; overflow: hidden; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.45); }
  .sq {
    width: 52px; height: 52px; position: relative;
    background: #4f6a3d; filter: saturate(0.35) brightness(0.55);
    transition: filter 350ms ease;
  }
  .sq.light { background: #b9b48f; }
  .sq.done, .sq.off { filter: none; }
  .sq.off { filter: saturate(0.2) brightness(0.7); }
  .sq.loading { animation: pulse 1.1s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { filter: saturate(0.35) brightness(0.55); } 50% { filter: saturate(0.9) brightness(1.05); } }
  .sq img {
    position: absolute; inset: 4px; width: 44px; height: 44px;
    animation: land 320ms cubic-bezier(0.2, 1.4, 0.4, 1);
    filter: drop-shadow(0 2px 2px rgba(0, 0, 0, 0.4));
  }
  .sq.off img { opacity: 0.35; filter: grayscale(1); }
  @keyframes land { from { transform: translateY(-14px) scale(0.6); opacity: 0; } to { transform: none; opacity: 1; } }
  .failed .sq { opacity: 0.4; }

  .line { margin: 24px 0 0; font-size: 17px; color: #e6e5e3; min-height: 20px; display: flex; gap: 10px; align-items: baseline; }
  .secs { color: #6f6d6a; font-size: 14px; font-variant-numeric: tabular-nums; }
  button {
    margin-top: 18px; padding: 8px 20px;
    background: #f2b441; color: #1b1608; border: 0; border-radius: 6px;
    font: inherit; font-size: 15px; font-weight: 600; cursor: pointer;
  }
</style>
