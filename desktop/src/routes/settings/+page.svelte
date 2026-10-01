<script lang="ts">
  // The settings window: every field of sidecoach.toml, generated from the pydantic
  // schema the sidecar serves (descriptions, choices, ranges, and when a change applies),
  // the API keys into the OS credential store, the audio devices by name, and the
  // shell's own two switches (what closing the window does, start with the system).
  // Opened at a setting (`#profile.chesscom_user`, from another window's link), its tab shows
  // and the field lights up; a save sends what changed and tells the other windows, and what
  // another window saved comes in here too.
  import { onDestroy, onMount, tick } from "svelte";
  import { connect, del, get, post, put } from "$lib/api";
  import { announce, inShell, onAnnounce, setShellSettings, shellSettings, type ShellSettings } from "$lib/shell.svelte";
  import { voiceLabel } from "$lib/voices";

  type Prop = { type?: string; anyOf?: { type: string }[]; description?: string; default?: unknown; choices?: string[]; choice_labels?: Record<string, string>; applies?: string; kind?: string; label?: string; section?: string; only_for?: string[]; advanced?: boolean; dev?: boolean; minimum?: number; maximum?: number; title?: string };
  type Described = {
    settings: Record<string, any>;
    schema: { properties: Record<string, any>; $defs: Record<string, { properties: Record<string, Prop>; description?: string }> };
    path: string;
    secrets: Record<string, boolean>;
    keyring: boolean;
    key_envs: Record<string, string>;
    devices: { inputs: { index: number; name: string }[]; outputs: { index: number; name: string }[] };
    voices: string[];
    personas: Record<string, string>;
    looks?: Looks;
  };
  // The boards' look (looks.py): the shipped sets with their authors and licences, the colour
  // themes, and the player's own pieces as they are.
  type Theme = { name: string; label: string; light: string; dark: string };
  type Looks = {
    sets: { name: string; label: string; author: string; licence: string; images: Record<string, string> }[];
    themes: Theme[];
    custom: { images: Record<string, string>; missing: string[] };
  };

  let data = $state<Described | null>(null);
  let values = $state<Record<string, any>>({});
  let secrets = $state<Record<string, string>>({});
  let clearSecrets = $state<Record<string, boolean>>({});
  let shell = $state<ShellSettings>({ close: "quit", autostart: false });
  // What the shell held before this save, so the line under Save can count the shell's own
  // settings (closing the window, start with the system) among what changed.
  let shellSaved = $state<ShellSettings>({ close: "quit", autostart: false });
  let advanced = $state(false);
  let saving = $state(false);
  let result = $state<{ ok: boolean; changed?: string[]; shellChanged?: boolean; now?: string[]; next?: string[]; restart?: string[]; errors?: { loc: (string | number)[]; msg: string }[] } | null>(null);
  let section = $state("llm");

  const SECTIONS: [string, string][] = [
    ["llm", "AI Coach"], ["coach", "Coach"], ["voice", "Voice"], ["engine", "Analysis"],
    ["capture", "Screen"], ["profile", "You"], ["board", "Board"], ["gpu", "GPU"], ["app", "App"],
  ];
  // Only the tabs this machine has: the phone's schema leaves out what it does not honour
  // (its GPU, its folders), and a tab with nothing in it is not shown.
  const shownSections = $derived(
    data
      ? SECTIONS.filter(([id]) => (id === "app" ? inShell() || !!data!.schema.properties.data_dir : !!data!.schema.properties[id]))
      : SECTIONS,
  );
  // The AI Coach page shows the provider, then only what that provider needs: a field's
  // only_for (from the schema) names the providers it is shown for. The key for a custom
  // service is stored under one fixed name.
  const CUSTOM_KEY = "CUSTOM_API_KEY";
  const providerNow = $derived(String(values.llm?.provider ?? "openai"));
  const keyName = $derived.by(() => {
    if (!data) return null;
    if (providerNow === "custom") return String(values.llm?.api_key_env || CUSTOM_KEY);
    return data.key_envs[providerNow] ?? null;
  });
  let listening = $state("");
  // The live test of the microphone and speech recognition under "Hearing you": while it is on the
  // page polls the level and what was heard; the coach is not asked.
  type Heard = { raw: string; heard: string; seconds: number; stt_ms: number; t: number };
  // Hardware acceleration, as this machine has it: shown under the setting only when a
  // graphics card is found -- whether CUDA is ready, the card, its memory and what the
  // app holds on it (the models, once loaded). Polled while the page is open.
  type GpuStatus = {
    mode: string;
    gpu: { found: boolean; eligible: boolean; reason: string; name?: string; driver?: string; vram?: number; vram_used?: number; vram_free?: number; used_by_app?: number | null };
    installed: string | null;
    active: { pack: string | null; cuda: boolean; detail: string };
  };
  let gpuStatus = $state<GpuStatus | null>(null);
  let gpuTimer: ReturnType<typeof setInterval> | null = null;
  async function pollGpu() {
    try {
      const st = await get<{ gpu?: GpuStatus }>("/api/setup/status");
      gpuStatus = st.gpu ?? null;
    } catch { /* the next poll will tell */ }
  }
  const gb = (n: number | undefined | null) => (n == null ? "–" : (n / 1024 ** 3).toFixed(1) + " GB");
  let hearOn = $state(false);
  let hearLevel = $state(0);
  let hearList = $state<Heard[]>([]);
  let hearError = $state("");
  let hearInfo = $state("");
  let hearTimer: ReturnType<typeof setInterval> | null = null;
  async function pollHearing() {
    try {
      const res = await get<{ on: boolean; level: number; heard: Heard[]; microphone: string; speech_recognition: string }>("/api/hear_test");
      hearLevel = res.level;
      hearList = res.heard;
      hearInfo = res.microphone;
      if (!res.on) stopHearing(false);
    } catch { /* the next poll will tell */ }
  }
  async function startHearing() {
    hearError = "";
    try {
      const res = await post<{ ok: boolean; error?: string }>("/api/hear_test", { on: true });
      if (!res.ok) { hearError = res.error ?? "could not start"; return; }
      hearOn = true;
      hearList = [];
      hearTimer = setInterval(pollHearing, 150);
    } catch (e) {
      hearError = String(e);
    }
  }
  async function stopHearing(tell = true) {
    if (hearTimer) clearInterval(hearTimer);
    hearTimer = null;
    hearOn = false;
    hearLevel = 0;
    if (tell) await post("/api/hear_test", { on: false }).catch(() => {});
  }
  onDestroy(() => { if (hearOn) void stopHearing(); if (gpuTimer) clearInterval(gpuTimer); });

  function fields(name: string): [string, Prop][] {
    if (!data) return [];
    const ref = data.schema.properties[name]?.$ref;
    if (!ref) return [];
    const model = data.schema.$defs[ref.split("/").pop()!];
    // A development field never shows in the app, and in the browser only under advanced.
    return Object.entries(model?.properties ?? {}).filter(([, p]) =>
      (advanced || !p.advanced) && !(p.dev && (inShell() || !advanced)) && (!p.only_for || p.only_for.includes(providerNow)));
  }
  // The headings on each tab, in the order they read; a field's section comes from the
  // schema. A heading with nothing under it (advanced off) is not shown.
  const SECTION_ORDER: Record<string, string[]> = {
    llm: ["Which model", "Speed and cost", "Development"],
    coach: ["When the coach speaks", "How the coach speaks", "What counts as a mistake", "A second opinion"],
    voice: ["The coach's voice", "Hearing you", "Under the hood"],
    engine: ["How hard Stockfish works", "What it looks at", "The program"],
    capture: ["Reading the screen"],
    profile: ["Your rating"],
    gpu: ["Hardware acceleration"],
  };
  function sections(name: string, list: [string, Prop][]): [string, [string, Prop][]][] {
    const order = SECTION_ORDER[name] ?? [];
    const seen = [...new Set(list.map(([, p]) => p.section ?? ""))];
    const titles = [...order.filter((t) => seen.includes(t)), ...seen.filter((t) => !order.includes(t))];
    return titles.map((t) => [t, list.filter(([, p]) => (p.section ?? "") === t)]);
  }
  /** A field's name as a person reads it. */
  function labelOf(key: string, p: Prop): string {
    return p.label ?? key.replaceAll("_", " ");
  }
  function choiceLabel(p: Prop, c: string): string {
    return p.choice_labels?.[c] ?? (c === "" ? "(default)" : c);
  }
  /** The Listen button: one line in the voice being tried; the coach's own voice stays. */
  async function listen(voice: string) {
    listening = voice;
    try {
      const res = await post<{ ok: boolean }>("/api/say", { voice });
      if (!res.ok) window.alert("The voice cannot speak yet: run Set up (Help menu) to fetch it, or check the Voice settings.");
    } catch (e) {
      window.alert("Could not play the voice: " + String(e));
    } finally {
      setTimeout(() => (listening = ""), 1500);
    }
  }
  function typeOf(p: Prop): string {
    if (p.type) return p.type;
    const t = (p.anyOf || []).map((a) => a.type).filter((t) => t !== "null");
    return t[0] ?? "string";
  }
  function isNullable(p: Prop): boolean {
    return !!(p.anyOf || []).some((a) => a.type === "null");
  }
  const secretNames = $derived.by(() => (keyName ? [keyName] : []) as string[]);
  /** Save, then restart the app (the shell) -- for the few settings that need it. */
  async function saveAndRestart() {
    if (!window.confirm("Save the settings and restart Sidecoach now?")) return;
    await save();
    if (!result?.ok) return;
    if (!inShell()) { window.alert("Saved. Restart the panel (sidecoach panel) for it to take effect."); return; }
    try {
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (e) {
      window.alert("Could not restart: " + String(e));
    }
  }
  /** "voice.voice" -> "Voice"; "voice.tts_device" -> "TTS device"; "llm.provider" -> "Provider". */
  function fieldName(path: string): string {
    const key = path.split(".").pop() ?? path;
    const words = key.split("_").map((w) => (["tts", "stt", "llm", "gpu", "url", "api", "ptt"].includes(w) ? w.toUpperCase() : w));
    const text = words.join(" ");
    return text[0].toUpperCase() + text.slice(1);
  }
  /** The line under Save: applied is the default and goes unsaid; what waits for a Start
   * or a restart is named. */
  function savedLine(r: NonNullable<typeof result>): string {
    const parts: string[] = [];
    if (r.next?.length) parts.push(`${r.next.map(fieldName).join(", ")}: at the next Start.`);
    if (r.restart?.length) parts.push(`${r.restart.map(fieldName).join(", ")}: after a restart.`);
    if (!r.changed?.length && !r.shellChanged) return "Saved. Nothing changed.";
    return parts.length ? "Saved. " + parts.join(" ") : "Saved.";
  }

  // The settings as saved, field by field: what the page has that differs is what it sends.
  let saved: Record<string, any> = {};

  /** Each field of `now` that differs from `before`, section by section. */
  function changes(now: Record<string, any>, before: Record<string, any>): Record<string, any> {
    const out: Record<string, any> = {};
    for (const [key, value] of Object.entries(now)) {
      const was = before[key];
      if (value && typeof value === "object" && !Array.isArray(value) && was && typeof was === "object") {
        const inner = changes(value, was);
        if (Object.keys(inner).length) out[key] = inner;
      } else if (JSON.stringify(value) !== JSON.stringify(was)) {
        out[key] = value;
      }
    }
    return out;
  }

  /** What another window saved, taken in: every field this page has not changed itself. */
  async function refresh(): Promise<void> {
    const fresh = await get<Described>("/api/settings").catch(() => null);
    if (!fresh) return;
    const mine = changes($state.snapshot(values), saved);
    data = fresh;
    const next = structuredClone(fresh.settings);
    for (const [sec, fields] of Object.entries(mine)) {
      if (fields && typeof fields === "object" && next[sec]) Object.assign(next[sec], fields);
      else next[sec] = fields;
    }
    values = next;
    saved = structuredClone(fresh.settings);
  }

  // A setting asked for by another window (`#profile.chesscom_user`): its tab, the field in
  // view, focused, and lit for a moment.
  async function reveal(): Promise<void> {
    const place = decodeURIComponent(location.hash.slice(1));
    const [sec, key] = place.split(".");
    if (!data || !sec || !key || !shownSections.some(([id]) => id === sec)) return;
    section = sec;
    if (data.schema.$defs && fields(sec).every(([k]) => k !== key)) advanced = true;
    await tick();
    const input = document.getElementById(`${sec}-${key}`);
    const box = input?.closest(".field");
    if (!input || !box) return;
    box.scrollIntoView({ block: "center", behavior: "smooth" });
    input.focus({ preventScroll: true });
    box.classList.remove("flash");
    void (box as HTMLElement).offsetWidth; // the animation starts over when asked again
    box.classList.add("flash");
  }

  onMount(() => {
    void pollGpu();
    gpuTimer = setInterval(() => void pollGpu(), 5000);
    const stop = onAnnounce("settings", () => void refresh());
    const onFocus = () => void refresh();
    const onHash = () => void reveal();
    window.addEventListener("focus", onFocus);
    window.addEventListener("hashchange", onHash);
    (async () => {
      await connect();
      data = await get<Described>("/api/settings");
      values = structuredClone($state.snapshot(data).settings);
      saved = structuredClone($state.snapshot(data).settings);
      shell = await shellSettings();
      shellSaved = structuredClone($state.snapshot(shell));
      await reveal();
    })();
    return () => {
      stop();
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("hashchange", onHash);
    };
  });

  function setValue(sec: string, key: string, p: Prop, raw: string | boolean) {
    const t = typeOf(p);
    let v: unknown = raw;
    if (t === "boolean") v = !!raw;
    else if (t === "integer") v = raw === "" ? (isNullable(p) ? null : values[sec][key]) : parseInt(String(raw), 10);
    else if (t === "number") v = raw === "" ? (isNullable(p) ? null : values[sec][key]) : parseFloat(String(raw));
    else v = raw === "" && isNullable(p) ? null : raw;
    values[sec][key] = v;
    result = null; // "Saved." is about the last save, not this change
  }

  async function save() {
    if (!data) return;
    saving = true;
    result = null;
    const sec: Record<string, string | null> = {};
    for (const name of Object.keys({ ...secrets, ...clearSecrets })) {
      if (clearSecrets[name]) sec[name] = null;
      else if (secrets[name]) sec[name] = secrets[name];
    }
    // A custom service's key lives under one name; the settings say which.
    if (values.llm && values.llm.provider === "custom" && !values.llm.api_key_env) values.llm.api_key_env = CUSTOM_KEY;
    try {
      // What changed here, and only that: a field another window saved since stays as it
      // saved it.
      result = await put("/api/settings", { settings: changes($state.snapshot(values), saved), secrets: sec });
      if (result?.ok) {
        const shellChanged = shell.close !== shellSaved.close || shell.autostart !== shellSaved.autostart;
        await setShellSettings(shell);
        shellSaved = structuredClone($state.snapshot(shell));
        result = { ...result, shellChanged };
        data = await get<Described>("/api/settings");
        values = structuredClone($state.snapshot(data).settings);
        saved = structuredClone($state.snapshot(data).settings);
        secrets = {};
        clearSecrets = {};
        void announce("settings");
      }
    } finally {
      saving = false;
    }
  }
  // ---- the Board tab: the pieces and the colours, shown as they would look ----
  // A preview row: the white king, queen and knight, the black bishop, rook and pawn, on the
  // squares of the colours chosen now.
  const PREVIEW = ["wK", "wQ", "wN", "bB", "bR", "bP"];
  const CODES = ["wK", "wQ", "wR", "wB", "wN", "wP", "bK", "bQ", "bR", "bB", "bN", "bP"];
  const PIECE_NAMES: Record<string, string> = { K: "king", Q: "queen", R: "rook", B: "bishop", N: "knight", P: "pawn" };
  const pieceName = (code: string) => `${code[0] === "w" ? "white" : "black"} ${PIECE_NAMES[code[1]]}`;
  const themeNow = $derived<Theme | null>(data?.looks?.themes.find((t) => t.name === values.board?.theme) ?? data?.looks?.themes[0] ?? null);
  const ownImages = $derived(data?.looks?.custom.images ?? {});
  const ownCount = $derived(Object.keys(ownImages).length);
  /** A piece of the player's own set, else Sidecoach's in its place (as the boards do). */
  const ownOr = (code: string) => ownImages[code] ?? data?.looks?.sets[0]?.images[code] ?? "";

  let loading = $state(false);
  let loaded = $state<{ taken: string[]; skipped: { name: string; why: string }[]; missing: string[]; error?: string } | null>(null);
  let fileInput = $state<HTMLInputElement | null>(null);

  function base64(buf: ArrayBuffer): string {
    const bytes = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  /** The images the player chose, sent to be kept in their data folder; the set picked. */
  async function loadOwn(files: FileList | null): Promise<void> {
    if (!files?.length) return;
    loading = true;
    loaded = null;
    try {
      const sent = await Promise.all([...files].map(async (f) => ({ name: f.name, data: base64(await f.arrayBuffer()) })));
      const res = await post<{ ok: boolean; taken: string[]; skipped: { name: string; why: string }[]; missing: string[] }>("/api/pieces/custom", { files: sent });
      loaded = res;
      await refresh();
      if (res.taken.length && values.board) {
        values.board.pieces = "custom";
        result = null;
      }
    } catch (e) {
      loaded = { taken: [], skipped: [], missing: [], error: "Could not load them: " + String(e) };
    } finally {
      loading = false;
      if (fileInput) fileInput.value = "";
    }
  }

  async function removeOwn(): Promise<void> {
    await del("/api/pieces/custom").catch(() => null);
    loaded = null;
    await refresh();
    if (values.board?.pieces === "custom") values.board.pieces = "sidecoach";
  }

  function errorFor(sec: string, key: string): string | null {
    const e = result?.errors?.find((x) => x.loc[0] === sec && x.loc[1] === key);
    return e ? e.msg : null;
  }
</script>

<svelte:head><title>Sidecoach · Settings</title></svelte:head>

{#snippet previewRow(src: (code: string) => string)}
  <span class="preview" aria-hidden="true">
    {#each PREVIEW as c, i (c)}
      <span class="pv" style:background={i % 2 ? themeNow?.dark : themeNow?.light}><img src={src(c)} alt="" draggable="false" /></span>
    {/each}
  </span>
{/snippet}

{#snippet field(section: string, key: string, p: Prop)}
  {@const t = typeOf(p)}
  {@const v = values[section]?.[key]}
  {@const err = errorFor(section, key)}
  <div class="field" class:error={!!err}>
    {#if t === "boolean"}
      <div class="row">
        <input id="{section}-{key}" type="checkbox" checked={!!v} onchange={(e) => setValue(section, key, p, e.currentTarget.checked)} />
        <label for="{section}-{key}">{labelOf(key, p)}</label>
      </div>
    {:else}
      <label for="{section}-{key}">{labelOf(key, p)}</label>
      {#if p.kind === "input_device" || p.kind === "output_device"}
        <select id="{section}-{key}" onchange={(e) => setValue(section, key, p, e.currentTarget.value)}>
          <option value="" selected={v == null || v === ""}>system default</option>
          {#each (p.kind === "input_device" ? data!.devices.inputs : data!.devices.outputs) as d (d.index)}
            <option value={d.name} selected={String(v) === d.name || String(v) === String(d.index)}>{d.name}</option>
          {/each}
        </select>
      {:else if p.kind === "voice"}
        <div class="row">
          <select id="{section}-{key}" onchange={(e) => setValue(section, key, p, e.currentTarget.value)}>
            {#each data!.voices as name (name)}<option value={name} selected={String(v) === name}>{voiceLabel(name)}</option>{/each}
            {#if v && !data!.voices.includes(String(v))}<option value={String(v)} selected>{voiceLabel(String(v))}</option>{/if}
          </select>
          <button type="button" class="listen" onclick={() => listen(String(v ?? ""))} disabled={!!listening} title="Hear a short line in this voice">{listening ? "Playing…" : "Listen"}</button>
        </div>
      {:else if p.kind === "persona"}
        <select id="{section}-{key}" onchange={(e) => setValue(section, key, p, e.currentTarget.value)}>
          {#each p.choices ?? [] as c (c)}<option value={c} selected={String(v) === c}>{data!.personas[c] ?? c}</option>{/each}
        </select>
      {:else if p.kind === "pieces" && data?.looks}
        <div id="{section}-{key}" class="sets" role="radiogroup" aria-label={labelOf(key, p)} tabindex="-1">
          {#each data.looks.sets as s (s.name)}
            <button type="button" class="set" role="radio" aria-checked={v === s.name} title="{s.label}: {s.author}, {s.licence}" onclick={() => setValue(section, key, p, s.name)}>
              {@render previewRow((c) => s.images[c])}
              <span class="set-name">{s.label}</span>
              <span class="set-by">{s.author === "Sidecoach" ? "Sidecoach's own" : `${s.author} · ${s.licence}`}</span>
            </button>
          {/each}
          <button type="button" class="set" role="radio" aria-checked={v === "custom"} disabled={!ownCount} title={ownCount ? "Your own pieces" : "Load your own pieces below first"} onclick={() => setValue(section, key, p, "custom")}>
            {#if ownCount}{@render previewRow(ownOr)}{:else}<span class="set-empty">Your images</span>{/if}
            <span class="set-name">Your own</span>
            <span class="set-by">{ownCount ? (ownCount === 12 ? "all 12 pieces" : `${ownCount} of 12 pieces`) : "load them below"}</span>
          </button>
        </div>
      {:else if p.kind === "board_theme" && data?.looks}
        <div id="{section}-{key}" class="themes" role="radiogroup" aria-label={labelOf(key, p)} tabindex="-1">
          {#each data.looks.themes as t (t.name)}
            <button type="button" class="theme" role="radio" aria-checked={v === t.name} title={t.label} onclick={() => setValue(section, key, p, t.name)}>
              <span class="swatch" aria-hidden="true"><i style:background={t.light}></i><i style:background={t.dark}></i><i style:background={t.dark}></i><i style:background={t.light}></i></span>
              <span class="theme-name">{t.label}</span>
            </button>
          {/each}
        </div>
      {:else if p.kind === "color"}
        <div class="row">
          <input id="{section}-{key}" class="color" type="color" value={String(v ?? p.default ?? "#ffffff")} oninput={(e) => setValue(section, key, p, e.currentTarget.value)} />
          <span class="hex">{String(v ?? p.default ?? "").toUpperCase()}</span>
        </div>
      {:else if p.choices}
        <select id="{section}-{key}" onchange={(e) => setValue(section, key, p, e.currentTarget.value)}>
          {#each p.choices as c (c)}<option value={c} selected={(v == null ? "" : String(v)) === c}>{choiceLabel(p, c)}</option>{/each}
        </select>
      {:else if t === "integer" || t === "number"}
        <input id="{section}-{key}" type="number" value={v ?? ""} min={p.minimum} max={p.maximum} step={t === "integer" ? 1 : "any"} oninput={(e) => setValue(section, key, p, e.currentTarget.value)} />
      {:else if t === "object"}
        <input id="{section}-{key}" type="text" value={JSON.stringify(v ?? {})} onchange={(e) => { try { values[section][key] = JSON.parse(e.currentTarget.value || "{}"); } catch { /* keep typing */ } }} />
      {:else}
        <input id="{section}-{key}" type="text" value={v ?? ""} placeholder={isNullable(p) ? "(unset)" : ""} oninput={(e) => setValue(section, key, p, e.currentTarget.value)} />
      {/if}
    {/if}
    <p class="desc">{p.description}{#if p.applies === "next"} <span class="applies next">· takes effect at the next Start</span>{:else if p.applies === "restart"} <span class="applies restart">· needs a <button type="button" class="link" onclick={saveAndRestart} title="Save the settings and restart Sidecoach">restart</button></span>{/if}</p>
    {#if err}<p class="err">{err}</p>{/if}
  </div>
{/snippet}

<div class="settings">
  <nav class="secnav">
    {#each shownSections as [id, label] (id)}
      <button class="secbtn" class:on={section === id} onclick={() => (section = id)}>{label}</button>
    {/each}
    <label class="ctl adv"><input type="checkbox" bind:checked={advanced} /> advanced</label>
  </nav>

  <div class="secbody">
    {#if !data}
      <p class="muted">Loading…</p>
    {:else if section === "app"}
      <h2>App</h2>
      {#if inShell()}
        <div class="field">
          <label for="close">When the window is closed</label>
          <select id="close" bind:value={shell.close}>
            <option value="quit">Quit</option>
            <option value="tray">Keep running in the tray (the coach keeps watching; the icon by the clock brings the window back)</option>
          </select>
        </div>
      {:else}
        <p class="muted">The shell's own settings show in the desktop app.</p>
      {/if}
      {#if data.schema.properties.data_dir}
        <div class="field">
          <label for="data_dir">Data folder</label>
          <input id="data_dir" type="text" value={values.data_dir} disabled={inShell()} oninput={(e) => (values.data_dir = e.currentTarget.value)} />
          <p class="desc">{data.schema.properties.data_dir?.description}{inShell() ? " The app decides where." : ""}</p>
        </div>
      {/if}
      {#if advanced && data.schema.properties.tools_dir}
        <div class="field">
          <label for="tools_dir">Tools folder</label>
          <input id="tools_dir" type="text" value={values.tools_dir} oninput={(e) => (values.tools_dir = e.currentTarget.value)} />
          <p class="desc">{data.schema.properties.tools_dir?.description}</p>
        </div>
      {/if}
      {#if data.path}<p class="muted">Settings file: <code>{data.path}</code></p>{/if}
    {:else}
      <h2>{SECTIONS.find(([id]) => id === section)?.[1]}</h2>
      {#if section === "llm"}
        {#each fields(section).filter(([k]) => k === "provider") as [key, p] (key)}{@render field(section, key, p)}{/each}
        {#if providerNow === "none"}
          <p class="desc lead">The coach speaks from the board and its templates: the word on every move, the lesson on a mistake, the questions and hints, the review. Free and offline.</p>
        {:else}
          {#if keyName}
            <div class="field">
              <label for="api-key">API key</label>
              <div class="secret">
                <span class="state">{data.secrets[keyName] ? "set" : "not set"}</span>
                <input id="api-key" type="password" placeholder={data.secrets[keyName] ? "replace…" : "paste the key…"} value={secrets[keyName] ?? ""} oninput={(e) => (secrets[keyName!] = e.currentTarget.value)} disabled={!data.keyring} />
                {#if data.secrets[keyName]}<label class="ctl"><input type="checkbox" bind:checked={clearSecrets[keyName]} /> forget</label>{/if}
              </div>
              <p class="desc">{data.keyring ? "Kept in the operating system's credential store, never in the settings file." : "No credential store is available here; the key comes from the environment or .env."}</p>
            </div>
          {/if}
          {#each sections(section, fields(section).filter(([k]) => k !== "provider")) as [title, list] (title)}
            {#if title && title !== "Which model"}<h3>{title}</h3>{/if}
            {#each list as [key, p] (key)}{@render field(section, key, p)}{/each}
          {/each}
        {/if}
      {:else}
        {@const groups = sections(section, fields(section))}
        {#each groups as [title, list] (title)}
          {#if title && groups.length > 1}<h3>{title}</h3>{/if}
          {#each list as [key, p] (key)}{@render field(section, key, p)}{/each}
          {#if section === "gpu" && gpuStatus?.gpu.found}
            <div class="field test">
              <span class="lbl">This machine</span>
              <p class="desc"><b>CUDA:</b> {gpuStatus.active.cuda ? "ready -- the models run on the graphics card" : gpuStatus.gpu.eligible ? (gpuStatus.installed ? "not active (" + gpuStatus.active.detail + ")" : "available -- the acceleration pack is not installed yet (Help > Set up)") : "not usable: " + gpuStatus.gpu.reason}</p>
              <p class="desc"><b>Graphics card:</b> {gpuStatus.gpu.name}{#if gpuStatus.gpu.driver} <span class="muted">(driver {gpuStatus.gpu.driver})</span>{/if}</p>
              <p class="desc"><b>Memory:</b> {gb(gpuStatus.gpu.vram)} in all, {gb(gpuStatus.gpu.vram_free)} free{#if gpuStatus.gpu.used_by_app != null}; Sidecoach is using {gb(gpuStatus.gpu.used_by_app)}{:else}; what Sidecoach uses is not reported by this driver{/if}</p>
            </div>
          {/if}
          {#if section === "board" && data.looks}
            <h3>Your own pieces</h3>
            <div class="field own">
              <p class="desc lead">Use any piece images you have: choose one image file a piece, PNG, SVG, WebP or JPEG. Name each file by its colour and piece: <code>wK</code> <code>wQ</code> <code>wR</code> <code>wB</code> <code>wN</code> <code>wP</code> for white, <code>bK</code> … <code>bP</code> for black (<code>white-king.png</code> works too). You can choose all twelve at once. They stay on this computer, in Sidecoach's data folder.</p>
              <div class="row">
                <button type="button" class="listen" disabled={loading} onclick={() => fileInput?.click()}>{loading ? "Loading…" : ownCount ? "Replace or add images…" : "Choose images…"}</button>
                {#if ownCount}<button type="button" class="link" onclick={removeOwn}>Remove them</button>{/if}
                <input bind:this={fileInput} class="hidden" type="file" multiple accept="image/png,image/svg+xml,image/webp,image/jpeg,.png,.svg,.webp,.jpg,.jpeg" onchange={(e) => loadOwn(e.currentTarget.files)} />
              </div>
              {#if loaded}
                {#if loaded.error}
                  <p class="err">{loaded.error}</p>
                {:else}
                  <p class="desc">{loaded.taken.length ? `Loaded ${loaded.taken.length} piece${loaded.taken.length === 1 ? "" : "s"}.` : "Nothing was loaded."}{#if loaded.missing.length && loaded.taken.length}{" "}Still missing: {loaded.missing.map(pieceName).join(", ")} (Sidecoach's are used for those).{/if}{#if loaded.taken.length}{" "}"Your own" is picked above: Save to use them.{/if}</p>
                  {#each loaded.skipped as s (s.name)}<p class="err">Not used: {s.name} -- {s.why}.</p>{/each}
                {/if}
              {/if}
              {#if ownCount}
                <div class="own-grid" aria-label="Your pieces">
                  {#each CODES as c, i (c)}
                    <span class="pv" class:gone={!ownImages[c]} title={ownImages[c] ? pieceName(c) : `${pieceName(c)}: not loaded`} style:background={(i + Math.floor(i / 6)) % 2 ? themeNow?.dark : themeNow?.light}><img src={ownOr(c)} alt={pieceName(c)} draggable="false" /></span>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
          {#if section === "voice" && title === "Hearing you"}
            <div class="field test">
              <span class="lbl">Test the microphone and speech recognition</span>
              <div class="row">
                <button type="button" class="listen" onclick={() => (hearOn ? stopHearing() : startHearing())}>{hearOn ? "Stop" : "Start"}</button>
                <div class="meter" title="microphone level"><div class="bar" style:width={`${Math.round(Math.min(1, hearLevel * 2.5) * 100)}%`}></div></div>
                {#if hearOn}<span class="muted">Say a move, like "knight to f3".</span>{/if}
              </div>
              {#if hearError}<p class="err">{hearError}</p>{/if}
              {#if hearOn && hearInfo}<p class="desc">Listening on {hearInfo}.</p>{/if}
              {#each hearList as h (h.t)}
                <p class="desc heard"><b>Heard:</b> {h.heard || h.raw || "(no words)"}{#if h.raw && h.heard && h.raw !== h.heard}&nbsp;<span class="muted">(heard as "{h.raw}"; corrected against the board)</span>{/if}&nbsp;<span class="muted">· {h.seconds} s, recognised in {h.stt_ms} ms</span></p>
              {/each}
              {#if !hearOn && !hearList.length && !hearError}<p class="desc">Start, speak, and see what is recognised as you go. The coach is not asked.</p>{/if}
            </div>
          {/if}
        {/each}
      {/if}
    {/if}
  </div>

  <footer class="secfoot">
    {#if !inShell()}
      <!-- Back where Settings was opened from (the panel, Your games' "Set it in Settings" on
           the phone); a tab of its own goes to the panel. -->
      <button type="button" class="link back" onclick={() => (history.length > 1 ? history.back() : (window.location.href = "/"))}>&larr; Back</button>
    {/if}
    {#if result?.ok}
      <span class="saved">{savedLine(result)}{#if result.restart?.length} <button type="button" class="link" onclick={saveAndRestart}>Restart now</button>{/if}</span>
    {:else if result && !result.ok}
      <span class="err">Not saved: {result.errors?.length} field{result.errors?.length === 1 ? "" : "s"} need attention.</span>
    {/if}
    <span class="spacer"></span>
    <button class="primary" onclick={save} disabled={saving || !data}>{saving ? "Saving…" : "Save"}</button>
  </footer>
</div>

<style>
  /* The screen, less the phone's bars when the page is drawn under them (the variables
     are nothing on a desktop), so the footer with Save stays on it. */
  .settings {
    display: grid; grid-template-rows: auto 1fr auto;
    height: 100vh; height: calc(100dvh - var(--inset-top, 0px) - var(--inset-bottom, 0px));
  }
  /* The tabs wrap: six of them side by side would be a floor of 600px, and a phone is
     412. `.adv` keeps to the end of whatever row it lands on. */
  .secnav { display: flex; flex-wrap: wrap; gap: 4px; padding: 10px 14px; border-bottom: 1px solid var(--line); align-items: center; }
  .secbtn { background: transparent; color: var(--muted); border: 1px solid transparent; border-radius: 6px; padding: 6px 10px; font: inherit; font-weight: 600; cursor: pointer; }
  .secbtn.on { color: var(--ink); background: var(--panel); border-color: var(--line); }
  .adv { margin-left: auto; color: var(--muted); font-size: 12px; }
  .secbody { overflow: auto; padding: 14px 18px; }
  h2 { margin: 0 0 12px; font-size: 15px; }
  .lead { margin: 0 0 10px; }
  h3 { margin: 22px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .listen { flex: 0 0 auto; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 6px 12px; font: inherit; font-weight: 600; cursor: pointer; }
  .listen:disabled { opacity: 0.6; cursor: default; }
  .field { margin: 0 0 14px; max-width: 720px; border-radius: 8px; }
  /* A setting another window asked for: it blinks, on and off three times over a second (Tai,
     2026-09-30: two soft pulses read as much less than that). Each blink is on for a sixth of
     a second, off for a sixth, with a short ramp between so it does not stutter. */
  .field:global(.flash) { animation: field-flash 1s linear; }
  @keyframes field-flash {
    0%, 14%, 33%, 47%, 66%, 80% { background: color-mix(in srgb, var(--accent) 30%, transparent); box-shadow: 0 0 0 8px color-mix(in srgb, var(--accent) 30%, transparent); }
    17%, 30%, 50%, 63%, 83%, 100% { background: transparent; box-shadow: 0 0 0 8px transparent; }
  }
  @media (prefers-reduced-motion: reduce) { .field:global(.flash) { animation-duration: 0.01s; } }
  .field label, .field .lbl { display: block; font-weight: 600; font-size: 13px; margin-bottom: 4px; }
  .field.test { margin-top: 4px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; }
  input.color { width: 56px; height: 34px; padding: 2px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); cursor: pointer; }
  .hex { font: 13px/1 ui-monospace, Consolas, monospace; color: var(--muted); }
  .meter { flex: 0 0 160px; height: 10px; background: var(--panel); border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }
  .meter .bar { height: 100%; background: var(--accent); transition: width 80ms linear; }
  .heard { margin-top: 6px; }
  .field .row { display: flex; align-items: center; gap: 8px; }
  .field .row label { margin: 0; }
  .field input[type="text"], .field input[type="number"], .field input[type="password"], .field select {
    width: 100%; box-sizing: border-box; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; font: inherit;
  }
  .field input[type="number"] { max-width: 180px; }
  .field.error input, .field.error select { border-color: var(--error); }
  .desc { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
  .applies { color: var(--muted); }
  .applies.restart { color: var(--warn); }
  .link { background: none; border: none; padding: 0; margin: 0; font: inherit; color: var(--accent); text-decoration: underline; cursor: pointer; }
  .link:hover { color: var(--ink); }
  .err { color: var(--error); font-size: 12px; margin: 4px 0 0; }
  .muted { color: var(--muted); font-size: 12px; }
  .secret { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 6px 0; }
  .secret input { min-width: 180px; }
  .secret .state { color: var(--muted); font-size: 12px; min-width: 50px; }
  .secret input { flex: 1; }
  .secfoot { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 10px 18px; border-top: 1px solid var(--line); }
  .saved { color: var(--accent); font-size: 13px; }
  .back { text-decoration: none; font-weight: 600; }
  .spacer { flex: 1; }
  code { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }

  /* The Board tab: a card a piece set (a row of its pieces on the colours chosen), a swatch a
     colour theme; the one chosen ringed in the accent. */
  .sets { display: grid; grid-template-columns: repeat(auto-fill, minmax(196px, 1fr)); gap: 8px; max-width: 900px; outline: none; }
  .set, .theme {
    display: flex; flex-direction: column; align-items: flex-start; gap: 3px; text-align: left;
    padding: 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); color: var(--ink);
    font: inherit; cursor: pointer;
  }
  .set:hover:not(:disabled), .theme:hover { border-color: var(--muted); }
  .set[aria-checked="true"], .theme[aria-checked="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
  .set:disabled { cursor: default; opacity: 0.6; }
  .preview { display: flex; border-radius: 4px; overflow: hidden; }
  .pv { width: 30px; height: 30px; display: grid; place-items: center; flex: none; }
  .pv img { width: 100%; height: 100%; display: block; }
  .set-name { font-weight: 700; font-size: 13px; }
  .set-by { color: var(--muted); font-size: 11px; line-height: 1.3; }
  .set-empty { width: 180px; height: 30px; display: grid; place-items: center; border: 1px dashed var(--line); border-radius: 4px; color: var(--muted); font-size: 12px; }
  .themes { display: grid; grid-template-columns: repeat(auto-fill, minmax(92px, 1fr)); gap: 8px; max-width: 900px; outline: none; }
  .theme { align-items: center; }
  .swatch { display: grid; grid-template-columns: 1fr 1fr; width: 48px; height: 48px; border-radius: 4px; overflow: hidden; }
  .theme-name { font-size: 12px; font-weight: 600; }
  .field.own { max-width: 900px; }
  .own-grid { display: grid; grid-template-columns: repeat(6, 44px); margin-top: 10px; border-radius: 4px; overflow: hidden; width: max-content; }
  .own-grid .pv { width: 44px; height: 44px; }
  .own-grid .pv.gone img { opacity: 0.35; }
  .hidden { display: none; }
</style>
