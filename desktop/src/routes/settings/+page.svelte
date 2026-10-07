<script lang="ts">
  // The settings window: every field of sidecoach.toml, generated from the pydantic
  // schema the sidecar serves (descriptions, choices, ranges, and when a change applies),
  // the API keys into the OS credential store, the audio devices by name, and the
  // shell's own two switches (what closing the window does, start with the system).
  // Opened at a setting (`#profile.chesscom_user`, from another window's link), its tab shows
  // and the field lights up. Every change is saved as it is made -- no Save button -- and the
  // save tells the other windows; what another window saved comes in here too.
  import { onDestroy, onMount, tick } from "svelte";
  import { SLOW_MS, connect, del, get, post, put } from "$lib/api";
  import { announce, inShell, onAnnounce, openWindow, setShellSettings, shellSettings, type ShellSettings } from "$lib/shell.svelte";
  import { voiceLabel } from "$lib/voices";
  import { keyFromEvent, keyFromMouse, keyName as hotkeyName, typesText } from "$lib/hotkey";
  import { saveQueue } from "$lib/savequeue";
  import "$lib/styles/review.css";

  type Prop = { type?: string; anyOf?: { type: string }[]; description?: string; default?: unknown; choices?: string[]; choice_labels?: Record<string, string>; applies?: string; kind?: string; label?: string; section?: string; only_for?: string[]; advanced?: boolean; dev?: boolean; kept?: boolean; minimum?: number; maximum?: number; title?: string };
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
  // themes, and the player's own sets, each with its pieces and what it lacks.
  type Theme = { name: string; label: string; light: string; dark: string };
  type Looks = {
    sets: { name: string; label: string; author: string; licence: string; images: Record<string, string> }[];
    themes: Theme[];
    own: { name: string; label: string; images: Record<string, string>; missing: string[] }[];
  };

  let data = $state<Described | null>(null);
  let values = $state<Record<string, any>>({});
  let secrets = $state<Record<string, string>>({});
  let clearSecrets = $state<Record<string, boolean>>({});
  let shell = $state<ShellSettings>({ close: "quit", autostart: false });
  // What the shell held at the last save: its own settings (closing the window, start with
  // the system) are saved when they differ.
  let shellSaved = $state<ShellSettings>({ close: "quit", autostart: false });
  let advanced = $state(false);
  let saving = $state(false);
  // The last save that went through, for the status line (what waits for a Start or a restart).
  let result = $state<{ ok: boolean; changed?: string[]; now?: string[]; next?: string[]; restart?: string[]; errors?: { loc: (string | number)[]; msg: string }[]; error?: string } | null>(null);
  // A field the sidecar refused ("llm.temperature" -> why): shown under it, and left out of the
  // saves until it is changed, so one bad value holds up nothing else.
  let fieldErrors = $state<Record<string, string>>({});
  // A save that did not get through at all (the sidecar not answering): said, and tried again.
  let failure = $state("");
  const refused = $derived(Object.keys(fieldErrors).length);
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

  // The push-to-talk key, set by pressing it: "Change" waits for the next key or mouse button
  // (Escape, or a click elsewhere, gives up), as a game's key bindings do.
  let capturing = $state<string | null>(null);
  let stopCapture = (): void => {};
  function capture(section: string, key: string, p: Prop, button: HTMLElement): void {
    const id = `${section}.${key}`;
    stopCapture();
    if (capturing === id) { capturing = null; return; }
    capturing = id;
    const take = (k: string | null) => { if (k) setValue(section, key, p, k); stopCapture(); };
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "Escape") stopCapture();
      else take(keyFromEvent(e));
    };
    const onDown = (e: MouseEvent) => {
      if (e.button === 0) { if (!button.contains(e.target as Node)) stopCapture(); return; }
      const k = keyFromMouse(e);
      if (k) { e.preventDefault(); take(k); }
    };
    // A side button's own work (back, forward) and the wheel's (scrolling) are kept from the page.
    const swallow = (e: MouseEvent) => { if (e.button > 0) e.preventDefault(); };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("mousedown", onDown, true);
    window.addEventListener("mouseup", swallow, true);
    window.addEventListener("auxclick", swallow, true);
    stopCapture = () => {
      capturing = null;
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("mousedown", onDown, true);
      // The button's release comes after the press that set it: let it be swallowed first.
      setTimeout(() => {
        window.removeEventListener("mouseup", swallow, true);
        window.removeEventListener("auxclick", swallow, true);
      }, 300);
      stopCapture = () => {};
    };
  }
  onDestroy(() => stopCapture());

  // The bar's back arrow, as every window's: the app's window closes, a browser tab goes back
  // (the panel, or Your games' "Fix it in Settings" on the phone). A change still being typed
  // is saved first.
  async function leave(): Promise<void> {
    await flush();
    window.close();
    if (history.length > 1) history.back();
    else window.location.href = "/";
  }

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
    // A development field never shows in the app, and in the browser only under advanced;
    // what the app keeps for the player (llm.models) is never a field.
    return Object.entries(model?.properties ?? {}).filter(([, p]) =>
      !p.kept && (advanced || !p.advanced) && !(p.dev && (inShell() || !advanced)) && (!p.only_for || p.only_for.includes(providerNow)));
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
      const res = await post<{ ok: boolean }>("/api/say", { voice }, { timeout: SLOW_MS });
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
  /** Restart the app (the shell), every change saved first -- for the few settings that need it. */
  async function restartNow() {
    if (!window.confirm("Restart Sidecoach now?")) return;
    await flush();
    if (failure || Object.keys(fieldErrors).length) return;
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
  /** The status line after a save: applied is the default and goes unsaid; what waits for a
   * Start or a restart is named. */
  function savedLine(r: NonNullable<typeof result>): string {
    const parts: string[] = [];
    if (r.next?.length) parts.push(`${r.next.map(fieldName).join(", ")}: at the next Start.`);
    if (r.restart?.length) parts.push(`${r.restart.map(fieldName).join(", ")}: after a restart.`);
    return parts.length ? "Saved. " + parts.join(" ") : "Saved.";
  }

  // The settings as saved, field by field: what the page has that differs is what it sends.
  let saved = $state.raw<Record<string, any>>({});

  /** Each field of `now` that differs from `before`, section by section; a field whole, an
   * object too, so a key taken out of one (llm.extra_body) goes with the save. */
  function changes(now: Record<string, any>, before: Record<string, any>, sections = true): Record<string, any> {
    const out: Record<string, any> = {};
    for (const [key, value] of Object.entries(now)) {
      const was = before[key];
      if (sections && value && typeof value === "object" && !Array.isArray(value) && was && typeof was === "object") {
        const inner = changes(value, was, false);
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
    // A change still being typed is saved when the window loses the focus, is hidden (the
    // phone's app switcher) or closes (its ✕, the app quitting): sent so as to outlive the page.
    const onBlur = () => void flush();
    const onHidden = () => { if (document.visibilityState === "hidden") void flush(true); };
    // The microphone test ends with the window: onDestroy does not run when it is closed, and
    // the test listened on (the server also ends one its page stopped polling).
    const onLeave = () => { void flush(true); if (hearOn) void stopHearing(); };
    window.addEventListener("focus", onFocus);
    window.addEventListener("hashchange", onHash);
    window.addEventListener("blur", onBlur);
    document.addEventListener("visibilitychange", onHidden);
    window.addEventListener("pagehide", onLeave);
    window.addEventListener("beforeunload", onLeave);
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
      window.removeEventListener("blur", onBlur);
      document.removeEventListener("visibilitychange", onHidden);
      window.removeEventListener("pagehide", onLeave);
      window.removeEventListener("beforeunload", onLeave);
      void flush(true);
    };
  });

  /** A field changed by the player: kept, and saved -- at once, or, for what is typed (`typed`),
   * once the typing pauses. */
  function setValue(sec: string, key: string, p: Prop, raw: string | boolean, typed = false) {
    const t = typeOf(p);
    let v: unknown = raw;
    if (t === "boolean") v = !!raw;
    else if (t === "integer") v = raw === "" ? (isNullable(p) ? null : values[sec][key]) : parseInt(String(raw), 10);
    else if (t === "number") v = raw === "" ? (isNullable(p) ? null : values[sec][key]) : parseFloat(String(raw));
    else v = raw === "" && isNullable(p) ? null : raw;
    if (typeof v === "number" && Number.isNaN(v)) return; // "-" on the way to "-1": not a number yet
    values[sec][key] = v;
    edited(`${sec}.${key}`, typed);
  }

  /** A change made: its old refusal goes, and it is saved (`typed`: once the typing pauses). */
  function edited(field: string, typed = false): void {
    if (field in fieldErrors) delete fieldErrors[field];
    if (typed) soon();
    else void flush();
  }

  // Every change is saved as it is made (Tai, 2026-10-01: a username typed into Settings was
  // lost when the window closed before Save; "save each change as you make it, with no Save
  // button"). One save at a time, in order (lib/savequeue.ts).
  const TYPING_MS = 600;
  let typing: ReturnType<typeof setTimeout> | null = null;
  const queue = saveQueue((keepalive) => saveOnce(keepalive));
  // This page, and its saves counted: the sidecar applies a save of the page only when it is
  // later than the one it kept last (the audit of 2026-10-02).
  const PAGE = typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  let saves = 0;

  /** Save once the typing pauses. */
  function soon(): void {
    if (typing) clearTimeout(typing);
    typing = setTimeout(() => {
      typing = null;
      void flush();
    }, TYPING_MS);
  }

  /** Save whatever differs from the saved settings now, after the save on its way if any;
   * `keepalive`: the page is closing, and the request must outlive it. */
  function flush(keepalive = false): Promise<void> {
    if (typing) {
      clearTimeout(typing);
      typing = null;
    }
    return queue.flush(keepalive);
  }

  /** The changes, less the fields the sidecar refused and nobody has changed since. */
  function sendable(all: Record<string, any>): Record<string, any> {
    const out: Record<string, any> = {};
    for (const [sec, value] of Object.entries(all)) {
      if (value && typeof value === "object" && !Array.isArray(value) && saved[sec] && typeof saved[sec] === "object") {
        const inner = Object.fromEntries(Object.entries(value).filter(([k]) => !(`${sec}.${k}` in fieldErrors)));
        if (Object.keys(inner).length) out[sec] = inner;
      } else if (!(sec in fieldErrors)) {
        out[sec] = value;
      }
    }
    return out;
  }

  async function saveOnce(keepalive: boolean): Promise<void> {
    if (!data) return;
    const sec: Record<string, string | null> = {};
    for (const name of Object.keys({ ...secrets, ...clearSecrets })) {
      if (clearSecrets[name]) sec[name] = null;
      else if (secrets[name]) sec[name] = secrets[name];
    }
    // A custom service's key lives under one name; the settings say which.
    if (values.llm && values.llm.provider === "custom" && !values.llm.api_key_env) values.llm.api_key_env = CUSTOM_KEY;
    const sent = $state.snapshot(values);
    // What changed here, and only that: a field another window saved since stays as it saved it.
    const body = sendable(changes(sent, saved));
    const shellChanged = inShell() && (shell.close !== shellSaved.close || shell.autostart !== shellSaved.autostart);
    if (!Object.keys(body).length && !Object.keys(sec).length && !shellChanged) return;
    saving = true;
    try {
      // The settings go before the shell's own: a window closing sends them with keepalive, and
      // the shell's call, awaited first, held them back until the page was gone (the audit of
      // 2026-10-02).
      const res =
        Object.keys(body).length || Object.keys(sec).length
          ? // Numbered: a save sent as the page closes goes beside the one on its way, and the
            // sidecar keeps the later one whichever lands first (settings_api `update`).
            await put<NonNullable<typeof result>>(
              "/api/settings",
              { settings: body, secrets: sec, page: PAGE, seq: ++saves },
              // A change can reload a model before the answer.
              { keepalive, timeout: SLOW_MS },
            )
          : null;
      if (shellChanged) {
        const now = structuredClone($state.snapshot(shell));
        await setShellSettings(now);
        shellSaved = now;
      }
      if (res && !res.ok) {
        const refused = res.errors ?? [];
        for (const e of refused) {
          const at = e.loc.filter((x) => typeof x === "string").slice(0, 2).join(".");
          fieldErrors[at || "settings"] = e.msg;
        }
        // The rest of the changes go on their own -- when the refused fields are out of them,
        // so a refusal the page cannot place is said, not sent again and again.
        const rest = sendable(changes(sent, saved));
        if (Object.keys(rest).length && JSON.stringify(rest) !== JSON.stringify(body)) queue.more();
        // A save the sidecar could not write (the disk) says why (the audit of 2026-10-02).
        failure = refused.length ? "" : (res.error ?? "Not saved: Sidecoach refused the change.");
        return;
      }
      if (res) {
        for (const name of Object.keys(sec)) {
          // The key as sent is in the store; one typed since is the next save's.
          if (secrets[name] === sec[name]) delete secrets[name];
          delete clearSecrets[name];
        }
        result = res;
        const fresh = await get<Described>("/api/settings");
        // What was changed here while this save was on its way stays as it is, for the next one.
        const mine = changes($state.snapshot(values), sent);
        const next = structuredClone(fresh.settings);
        for (const [s, fields] of Object.entries(mine)) {
          if (fields && typeof fields === "object" && !Array.isArray(fields) && next[s]) Object.assign(next[s], fields);
          else next[s] = fields;
        }
        data = fresh;
        values = next;
        saved = structuredClone(fresh.settings);
        void announce("settings");
      }
      failure = "";
    } catch (e) {
      failure = "Not saved: Sidecoach did not answer. The change is kept here and saved with the next one.";
    } finally {
      saving = false;
    }
  }
  // ---- the Board tab: the pieces and the colours, shown as they would look ----
  // A preview row: the white king, queen and knight, the black bishop, rook and pawn, on the
  // squares of the colours chosen now.
  const PREVIEW = ["wK", "wQ", "wN", "bB", "bR", "bP"];
  const PIECE_NAMES: Record<string, string> = { K: "king", Q: "queen", R: "rook", B: "bishop", N: "knight", P: "pawn" };
  const pieceName = (code: string) => `${code[0] === "w" ? "white" : "black"} ${PIECE_NAMES[code[1]]}`;
  const themeNow = $derived<Theme | null>(data?.looks?.themes.find((t) => t.name === values.board?.theme) ?? data?.looks?.themes[0] ?? null);
  // The player's own sets (looks.py): each a tile among the shipped ones, with a menu to add or
  // replace its images, rename it and delete it (an Undo for a while); "Add a set..." makes one
  // from the images chosen, picked, named "My set" with the name ready to type over, as a new
  // folder is (Tai, 2026-10-01: more than one set, and each one deletable).
  type OwnSet = Looks["own"][number];
  type OwnResult = { ok: boolean; id?: string; name?: string; taken: string[]; skipped: { name: string; why: string }[]; missing: string[]; error?: string };
  const ownSets = $derived(data?.looks?.own ?? []);
  /** A piece of an own set, else Sidecoach's in its place (as the boards do). */
  const ownOr = (o: OwnSet, code: string) => o.images[code] ?? data?.looks?.sets[0]?.images[code] ?? "";
  const ownId = (name: string) => name.slice("own:".length);
  /** The setting names this set: "own:<id>", or "custom", the one own set there was before. */
  const isPicked = (v: unknown, name: string) => v === name || (v === "custom" && name === "own:custom");

  let busy = $state(false);
  let ownNote = $state<{ text: string; errors: string[]; bad?: boolean } | null>(null);
  let fileInput = $state<HTMLInputElement | null>(null);
  let fileFor = $state<string | null>(null); // the set the images go to; null: a new one
  let menuFor = $state<string | null>(null);
  let renaming = $state<string | null>(null);
  let renameText = $state("");
  let undoSet = $state<{ id: string; label: string; wasSaved: boolean } | null>(null);
  let undoTimer: ReturnType<typeof setTimeout> | null = null;

  function base64(buf: ArrayBuffer): string {
    const bytes = new Uint8Array(buf);
    let s = "";
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  function choose(forSet: string | null): void {
    fileFor = forSet;
    menuFor = null;
    fileInput?.click();
  }

  function noteOf(res: OwnResult, added: boolean): { text: string; errors: string[]; bad?: boolean } {
    const errors = res.skipped.map((x) => `Not used: ${x.name} — ${x.why}.`);
    if (!res.ok) return { text: res.error ?? "Nothing was loaded: no image's name says which piece it is.", errors, bad: true };
    const n = res.taken.length;
    const what = `${n} piece${n === 1 ? "" : "s"}`;
    let text = added ? `A new set: ${what}.` : `${what} into "${res.name}".`;
    if (res.missing.length) text += ` Still missing: ${res.missing.map(pieceName).join(", ")} (Sidecoach's stand in).`;
    if (added) text += " It is picked.";
    return { text, errors };
  }

  async function chosen(files: FileList | null): Promise<void> {
    if (!files?.length) return;
    busy = true;
    ownNote = null;
    const into = fileFor;
    try {
      const sent = await Promise.all([...files].map(async (f) => ({ name: f.name, data: base64(await f.arrayBuffer()) })));
      const res = into
        ? // Read piece by piece: as long as the images take, the button says it is busy.
          await post<OwnResult>(`/api/pieces/own/${into}/images`, { files: sent }, { timeout: null })
        : await post<OwnResult>("/api/pieces/own", { name: "", files: sent }, { timeout: null });
      ownNote = noteOf(res, into === null);
      await refresh();
      if (res.ok && into === null && res.id && values.board) {
        values.board.pieces = res.id;
        edited("board.pieces");
        startRename(res.id, res.name ?? "");
      }
    } catch (e) {
      ownNote = { text: "Could not load them: " + String(e), errors: [], bad: true };
    } finally {
      busy = false;
      fileFor = null;
      if (fileInput) fileInput.value = "";
    }
  }

  function startRename(name: string, label: string): void {
    menuFor = null;
    renaming = name;
    renameText = label;
    void tick().then(() => {
      const box = document.getElementById("rename-" + ownId(name)) as HTMLInputElement | null;
      box?.focus();
      box?.select();
    });
  }

  async function finishRename(keep: boolean): Promise<void> {
    const name = renaming;
    renaming = null;
    if (!name || !keep) return;
    const text = renameText.trim();
    if (!text || text === ownSets.find((o) => o.name === name)?.label) return;
    const res = await put<{ ok: boolean; error?: string }>(`/api/pieces/own/${ownId(name)}`, { name: text }).catch(() => null);
    if (!res?.ok) ownNote = { text: res?.error ?? "Could not rename it.", errors: [], bad: true };
    await refresh();
  }

  /** The board setting set and saved: a set deleted (or back) is no set to keep pointing at. */
  async function savePieces(pieces: string): Promise<void> {
    await refresh();
    if (values.board) values.board.pieces = pieces;
    await flush();
  }

  async function removeSet(o: OwnSet): Promise<void> {
    menuFor = null;
    ownNote = null;
    const res = await del<{ ok: boolean; error?: string }>(`/api/pieces/own/${ownId(o.name)}`).catch(() => null);
    if (!res?.ok) {
      ownNote = { text: res?.error ?? "Could not delete it.", errors: [], bad: true };
      return;
    }
    const wasSaved = isPicked(saved.board?.pieces, o.name);
    undoSet = { id: ownId(o.name), label: o.label, wasSaved };
    if (wasSaved) await savePieces("sidecoach");
    else {
      await refresh();
      if (values.board && isPicked(values.board.pieces, o.name)) values.board.pieces = saved.board?.pieces ?? "sidecoach";
    }
    if (undoTimer) clearTimeout(undoTimer);
    undoTimer = setTimeout(() => (undoSet = null), 8000);
  }

  async function undoRemove(): Promise<void> {
    const u = undoSet;
    undoSet = null;
    if (undoTimer) clearTimeout(undoTimer);
    if (!u) return;
    const res = await post<{ ok: boolean; error?: string }>(`/api/pieces/own/${u.id}/restore`).catch(() => null);
    if (!res?.ok) {
      ownNote = { text: res?.error ?? "Could not put it back.", errors: [], bad: true };
      return;
    }
    if (u.wasSaved) await savePieces("own:" + u.id);
    else await refresh();
  }

  // A press anywhere but the menu closes it; Escape too.
  $effect(() => {
    if (!menuFor) return;
    const away = (e: PointerEvent) => {
      const at = e.target as HTMLElement;
      if (!at.closest(".set-menu") && !at.closest(".set-more")) menuFor = null;
    };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") menuFor = null; };
    window.addEventListener("pointerdown", away, true);
    window.addEventListener("keydown", esc);
    return () => {
      window.removeEventListener("pointerdown", away, true);
      window.removeEventListener("keydown", esc);
    };
  });

  function errorFor(sec: string, key: string): string | null {
    return fieldErrors[`${sec}.${key}`] ?? null;
  }
  /** A JSON field (llm.extra_body) as typed: saved when it reads, else said and kept back. */
  function setObject(sec: string, key: string, text: string): void {
    try {
      values[sec][key] = JSON.parse(text || "{}");
      edited(`${sec}.${key}`);
    } catch {
      fieldErrors[`${sec}.${key}`] = "Not valid JSON, so not saved: check the brackets, quotes and commas.";
    }
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
          <option value="" selected={v == null || v === ""}>System default</option>
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
      {:else if p.kind === "hotkey"}
        {@const armed = capturing === `${section}.${key}`}
        <div class="row hotkey">
          <kbd class="keycap" class:none={!v}>{hotkeyName(String(v ?? ""))}</kbd>
          <button
            type="button"
            id="{section}-{key}"
            class="listen"
            class:armed
            onclick={(e) => capture(section, key, p, e.currentTarget)}
            title={armed ? "Press the key or mouse button you want; Escape to keep the one you have" : "Choose the key by pressing it"}
          >{armed ? "Press a key or a mouse side button…" : "Change"}</button>
          {#if v && !armed}<button type="button" class="link" onclick={() => setValue(section, key, p, "")}>Turn off</button>{/if}
        </div>
        {#if typesText(String(v ?? ""))}<p class="note warn">{hotkeyName(String(v))} also types into the game while you hold it; a key like Right Ctrl does not.</p>{/if}
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
          {#each ownSets as o (o.name)}
            {@const count = 12 - o.missing.length}
            <div class="own-set">
              <button type="button" class="set" role="radio" aria-checked={isPicked(v, o.name)} title="{o.label}: your own, {count} of 12 pieces" onclick={() => setValue(section, key, p, o.name)}>
                {@render previewRow((c) => ownOr(o, c))}
                <span class="set-name" class:hiding={renaming === o.name}>{o.label}</span>
                <span class="set-by">Yours · {count === 12 ? "all 12 pieces" : `${count} of 12 pieces`}</span>
              </button>
              {#if renaming === o.name}
                <input
                  id="rename-{ownId(o.name)}"
                  class="rename"
                  bind:value={renameText}
                  maxlength="40"
                  aria-label="The set's name"
                  onkeydown={(e) => { if (e.key === "Enter") void finishRename(true); if (e.key === "Escape") void finishRename(false); }}
                  onblur={() => void finishRename(true)}
                />
              {/if}
              <button type="button" class="set-more" aria-label="{o.label}: more" aria-haspopup="menu" aria-expanded={menuFor === o.name} title="Add images, rename or delete" onclick={() => (menuFor = menuFor === o.name ? null : o.name)}>
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4zm6 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4zm6 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4z" /></svg>
              </button>
              {#if menuFor === o.name}
                <div class="set-menu" role="menu" aria-label={o.label}>
                  <button role="menuitem" onclick={() => choose(ownId(o.name))}>Add or replace images…</button>
                  <button role="menuitem" onclick={() => startRename(o.name, o.label)}>Rename</button>
                  <button role="menuitem" class="warn" onclick={() => void removeSet(o)}>Delete</button>
                </div>
              {/if}
            </div>
          {/each}
          <button type="button" class="set add-set" disabled={busy} title="Choose a set's images, one a piece" onclick={() => choose(null)}>
            <span class="add-plus" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z" /></svg></span>
            <span class="set-name">{busy ? "Loading…" : "Add a set…"}</span>
            <span class="set-by">Your own images</span>
          </button>
        </div>
        <input bind:this={fileInput} class="hidden" type="file" multiple accept="image/png,image/svg+xml,image/webp,image/jpeg,.png,.svg,.webp,.jpg,.jpeg" onchange={(e) => void chosen(e.currentTarget.files)} />
        <p class="desc own-help">Your own set: one image a piece, PNG, SVG, WebP or JPEG, each named by its colour and piece — <code>wK</code> <code>wQ</code> <code>wR</code> <code>wB</code> <code>wN</code> <code>wP</code>, <code>bK</code> … <code>bP</code> (<code>white-king.png</code> works too). They are copied into Sidecoach's data folder.</p>
        {#if ownNote}
          <p class="desc" class:err={ownNote.bad}>{ownNote.text}</p>
          {#each ownNote.errors as e (e)}<p class="err">{e}</p>{/each}
        {/if}
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
          <input id="{section}-{key}" class="color" type="color" value={String(v ?? p.default ?? "#ffffff")} oninput={(e) => setValue(section, key, p, e.currentTarget.value, true)} />
          <span class="hex">{String(v ?? p.default ?? "").toUpperCase()}</span>
        </div>
      {:else if p.choices}
        <select id="{section}-{key}" onchange={(e) => setValue(section, key, p, e.currentTarget.value)}>
          {#each p.choices as c (c)}<option value={c} selected={(v == null ? "" : String(v)) === c}>{choiceLabel(p, c)}</option>{/each}
        </select>
      {:else if t === "integer" || t === "number"}
        <input id="{section}-{key}" type="number" value={v ?? ""} min={p.minimum} max={p.maximum} step={t === "integer" ? 1 : "any"} oninput={(e) => setValue(section, key, p, e.currentTarget.value, true)} onchange={() => void flush()} />
      {:else if t === "object"}
        <input id="{section}-{key}" type="text" value={JSON.stringify(v ?? {})} onchange={(e) => setObject(section, key, e.currentTarget.value)} />
      {:else}
        <input id="{section}-{key}" type="text" value={v ?? ""} placeholder={isNullable(p) ? "(unset)" : ""} oninput={(e) => setValue(section, key, p, e.currentTarget.value, true)} onchange={() => void flush()} />
      {/if}
    {/if}
    <p class="desc">{p.description}{#if p.applies === "next"}{" "}<span class="applies next">· takes effect at the next Start</span>{:else if p.applies === "restart"}{" "}<span class="applies restart">· needs a <button type="button" class="link" onclick={restartNow} title="Restart Sidecoach">restart</button></span>{/if}</p>
    {#if err}<p class="err">{err}</p>{/if}
  </div>
{/snippet}

<div class="settings">
  <header id="bar">
    <div class="group">
      <button class="iconbtn rv-back" aria-label="Close Settings" title="Close Settings" onclick={leave}>
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
      </button>
      <img class="rv-logo" src="/logo.svg" alt="" draggable="false" />
      <div class="rv-name">Settings</div>
    </div>
  </header>
  <nav class="secnav">
    {#each shownSections as [id, label] (id)}
      <button class="secbtn" class:on={section === id} onclick={() => (section = id)}>{label}</button>
    {/each}
    <label class="ctl adv" title="Show the settings few people need"><input type="checkbox" bind:checked={advanced} /> Advanced</label>
  </nav>

  <div class="secbody">
    {#if !data}
      <p class="muted">Loading…</p>
    {:else if section === "app"}
      <h2>App</h2>
      {#if inShell()}
        <div class="field">
          <label for="close">When the window is closed</label>
          <select id="close" bind:value={shell.close} onchange={() => void flush()}>
            <option value="quit">Quit</option>
            <option value="tray">Keep running in the tray (the coach keeps watching; the icon by the clock brings the window back)</option>
          </select>
        </div>
      {:else}
        <p class="muted">What closing the window does is set in the desktop app.</p>
      {/if}
      {#if data.schema.properties.data_dir}
        <div class="field">
          <label for="data_dir">Data folder</label>
          <input id="data_dir" type="text" value={values.data_dir} disabled={inShell()} oninput={(e) => { values.data_dir = e.currentTarget.value; edited("data_dir", true); }} onchange={() => void flush()} />
          <p class="desc">{data.schema.properties.data_dir?.description}{inShell() ? " The app decides where." : ""}</p>
        </div>
      {/if}
      <!-- A development field, never shown in the app, whose shell keeps Stockfish in its own
        folder: there a change was taken and gone at the restart (the settings review, 2026-10-01). -->
      {#if advanced && !inShell() && data.schema.properties.tools_dir}
        <div class="field">
          <label for="tools_dir">Tools folder</label>
          <input id="tools_dir" type="text" value={values.tools_dir} oninput={(e) => { values.tools_dir = e.currentTarget.value; edited("tools_dir", true); }} onchange={() => void flush()} />
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
                <!-- Saved when it is left or Enter is pressed, not as it is typed: a key half
                     typed is no key. -->
                <input id="api-key" type="password" placeholder={data.secrets[keyName] ? "replace…" : "paste the key…"} value={secrets[keyName] ?? ""} oninput={(e) => (secrets[keyName!] = e.currentTarget.value)} onchange={() => void flush()} onkeydown={(e) => { if (e.key === "Enter") void flush(); }} disabled={!data.keyring} />
                {#if data.secrets[keyName]}<label class="ctl"><input type="checkbox" bind:checked={clearSecrets[keyName]} onchange={() => void flush()} /> forget</label>{/if}
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
              <p class="desc"><b>CUDA:</b>
                {#if gpuStatus.active.cuda}ready: the models run on the graphics card.
                {:else if gpuStatus.gpu.eligible && gpuStatus.installed}not active ({gpuStatus.active.detail}).
                {:else if gpuStatus.gpu.eligible}available, but its pack is not downloaded yet. <a class="link" href="/setup" target="_blank" rel="noopener" onclick={(e) => { if (inShell()) { e.preventDefault(); void openWindow("setup"); } }}>Set it up</a>
                {:else}not usable: {gpuStatus.gpu.reason}{/if}
              </p>
              <p class="desc"><b>Graphics card:</b> {gpuStatus.gpu.name}{#if gpuStatus.gpu.driver}{" "}<span class="muted">(driver {gpuStatus.gpu.driver})</span>{/if}</p>
              <p class="desc"><b>Memory:</b> {gb(gpuStatus.gpu.vram)} in all, {gb(gpuStatus.gpu.vram_free)} free{#if gpuStatus.gpu.used_by_app != null}; Sidecoach is using {gb(gpuStatus.gpu.used_by_app)}{:else}; what Sidecoach uses is not reported by this driver{/if}</p>
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

  {#if undoSet}
    <div class="gm-undo" role="status">
      <span>Deleted "{undoSet.label}"</span>
      <button type="button" class="gm-undo-btn" onclick={() => void undoRemove()}>Undo</button>
    </div>
  {/if}
  <!-- Each change saves itself; this line says how the last one went, and only when there is
       something to say. -->
  {#if saving || failure || refused || result?.ok}
    <footer class="secfoot" role="status">
      {#if saving}
        <span class="muted">Saving…</span>
      {:else if failure}
        <span class="err">{failure}</span>
        <button type="button" class="link" onclick={() => void flush()}>Try again</button>
      {:else if refused}
        <span class="err">{refused === 1 ? "One setting is" : `${refused} settings are`} not saved: see the note in red.</span>
      {:else if result?.ok}
        <span class="saved">{savedLine(result)}{#if result.restart?.length}{" "}<button type="button" class="link" onclick={restartNow}>Restart now</button>{/if}</span>
      {/if}
    </footer>
  {/if}
</div>

<style>
  /* The screen, less the phone's bars when the page is drawn under them (the variables
     are nothing on a desktop), so the status line stays on it. */
  .settings {
    display: grid; grid-template-rows: auto auto 1fr auto;
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
  /* The push-to-talk key: the key as a keycap, Change waiting for a press lit in the accent. */
  .keycap {
    min-width: 44px; padding: 5px 12px; border: 1px solid var(--line); border-bottom-width: 3px; border-radius: 6px;
    background: var(--panel2); color: var(--ink); font: inherit; font-weight: 700; text-align: center;
  }
  .keycap.none { color: var(--muted); font-weight: 500; }
  .listen.armed { border-color: var(--accent); color: var(--accent); box-shadow: 0 0 0 3px #81b64c33; }
  .note.warn { margin: 6px 0 0; font-size: 12px; color: var(--warn); }
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
  .themes { display: grid; grid-template-columns: repeat(auto-fill, minmax(92px, 1fr)); gap: 8px; max-width: 900px; outline: none; }
  .theme { align-items: center; }
  .swatch { display: grid; grid-template-columns: 1fr 1fr; width: 48px; height: 48px; border-radius: 4px; overflow: hidden; }
  .theme-name { font-size: 12px; font-weight: 600; }
  /* An own set: its tile, a round menu button in its corner, the menu below it; renamed in place. */
  .own-set { position: relative; display: grid; }
  .own-set .set { width: 100%; padding-right: 34px; }
  .set-name.hiding { visibility: hidden; }
  .set-more {
    position: absolute; top: 6px; right: 6px; width: 28px; height: 28px; padding: 0; border: 0; border-radius: 50%;
    display: grid; place-items: center; background: transparent; color: var(--muted); cursor: pointer;
  }
  .set-more svg { width: 20px; height: 20px; fill: currentColor; }
  .set-more:hover, .set-more[aria-expanded="true"] { background: var(--panel2); color: var(--ink); }
  .set-menu {
    position: absolute; top: 36px; right: 6px; z-index: 20; display: flex; flex-direction: column; min-width: 190px; padding: 6px;
    background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 10px 28px #000b;
  }
  .set-menu button {
    padding: 9px 10px; border: 0; border-radius: 6px; background: transparent; color: var(--ink);
    font: inherit; font-size: 14px; text-align: left; cursor: pointer; white-space: nowrap;
  }
  .set-menu button:hover, .set-menu button:focus-visible { background: var(--panel3); outline: none; }
  .set-menu .warn { color: var(--warn); }
  .rename {
    position: absolute; left: 6px; right: 38px; top: 40px; height: 24px; padding: 0 6px; font: inherit; font-size: 13px; font-weight: 700;
    color: var(--ink); background: var(--panel2); border: 1px solid var(--accent); border-radius: 5px; outline: none;
  }
  .add-set { border-style: dashed; justify-content: center; min-height: 82px; }
  .add-plus { width: 30px; height: 30px; display: grid; place-items: center; border-radius: 50%; background: var(--panel2); }
  .add-plus svg { width: 20px; height: 20px; fill: var(--accent); }
  .own-help { max-width: 900px; }
  .hidden { display: none; }
</style>
