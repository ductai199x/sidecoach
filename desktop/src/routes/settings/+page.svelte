<script lang="ts">
  // The settings window: every field of sidecoach.toml, generated from the pydantic
  // schema the sidecar serves (descriptions, choices, ranges, and when a change applies),
  // the API keys into the OS credential store, the audio devices by name, and the
  // shell's own two switches (what closing the window does, start with the system).
  import { onMount } from "svelte";
  import { connect, get, post, put } from "$lib/api";
  import { inShell, setShellSettings, shellSettings, type ShellSettings } from "$lib/shell.svelte";
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
  };

  let data = $state<Described | null>(null);
  let values = $state<Record<string, any>>({});
  let secrets = $state<Record<string, string>>({});
  let clearSecrets = $state<Record<string, boolean>>({});
  let shell = $state<ShellSettings>({ close: "tray", autostart: false });
  let advanced = $state(false);
  let saving = $state(false);
  let result = $state<{ ok: boolean; changed?: string[]; now?: string[]; next?: string[]; restart?: string[]; errors?: { loc: (string | number)[]; msg: string }[] } | null>(null);
  let section = $state("llm");

  const SECTIONS: [string, string][] = [
    ["llm", "AI Coach"], ["coach", "Coach"], ["voice", "Voice"], ["engine", "Analysis"],
    ["capture", "Screen"], ["profile", "You"], ["gpu", "GPU"], ["app", "App"],
  ];
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
    gpu: ["Graphics card"],
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
    if (!r.changed?.length) return "Saved. Nothing changed.";
    return parts.length ? "Saved. " + parts.join(" ") : "Saved.";
  }

  onMount(async () => {
    await connect();
    data = await get<Described>("/api/settings");
    values = structuredClone($state.snapshot(data).settings);
    shell = await shellSettings();
  });

  function setValue(sec: string, key: string, p: Prop, raw: string | boolean) {
    const t = typeOf(p);
    let v: unknown = raw;
    if (t === "boolean") v = !!raw;
    else if (t === "integer") v = raw === "" ? (isNullable(p) ? null : values[sec][key]) : parseInt(String(raw), 10);
    else if (t === "number") v = raw === "" ? (isNullable(p) ? null : values[sec][key]) : parseFloat(String(raw));
    else v = raw === "" && isNullable(p) ? null : raw;
    values[sec][key] = v;
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
      result = await put("/api/settings", { settings: values, secrets: sec });
      if (result?.ok) {
        await setShellSettings(shell);
        data = await get<Described>("/api/settings");
        secrets = {};
        clearSecrets = {};
      }
    } finally {
      saving = false;
    }
  }
  function errorFor(sec: string, key: string): string | null {
    const e = result?.errors?.find((x) => x.loc[0] === sec && x.loc[1] === key);
    return e ? e.msg : null;
  }
</script>

<svelte:head><title>Sidecoach · Settings</title></svelte:head>

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
    {#each SECTIONS as [id, label] (id)}
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
            <option value="tray">Keep running in the tray (the coach keeps watching; the icon by the clock brings the window back)</option>
            <option value="quit">Quit everything</option>
          </select>
        </div>
      {:else}
        <p class="muted">The shell's own settings show in the desktop app.</p>
      {/if}
      <div class="field">
        <label for="data_dir">Data folder</label>
        <input id="data_dir" type="text" value={values.data_dir} disabled={inShell()} oninput={(e) => (values.data_dir = e.currentTarget.value)} />
        <p class="desc">{data.schema.properties.data_dir?.description}{inShell() ? " The app decides where." : ""}</p>
      </div>
      {#if advanced}
        <div class="field">
          <label for="tools_dir">Tools folder</label>
          <input id="tools_dir" type="text" value={values.tools_dir} oninput={(e) => (values.tools_dir = e.currentTarget.value)} />
          <p class="desc">{data.schema.properties.tools_dir?.description}</p>
        </div>
      {/if}
      <p class="muted">Settings file: <code>{data.path}</code></p>
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
        {/each}
      {/if}
    {/if}
  </div>

  <footer class="secfoot">
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
  .settings { display: grid; grid-template-rows: auto 1fr auto; height: 100vh; }
  .secnav { display: flex; gap: 4px; padding: 10px 14px; border-bottom: 1px solid var(--line); align-items: center; }
  .secbtn { background: transparent; color: var(--muted); border: 1px solid transparent; border-radius: 6px; padding: 6px 10px; font: inherit; font-weight: 600; cursor: pointer; }
  .secbtn.on { color: var(--ink); background: var(--panel); border-color: var(--line); }
  .adv { margin-left: auto; color: var(--muted); font-size: 12px; }
  .secbody { overflow: auto; padding: 14px 18px; }
  h2 { margin: 0 0 12px; font-size: 15px; }
  .lead { margin: 0 0 10px; }
  h3 { margin: 22px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .listen { flex: 0 0 auto; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 6px 12px; font: inherit; font-weight: 600; cursor: pointer; }
  .listen:disabled { opacity: 0.6; cursor: default; }
  .field { margin: 0 0 14px; max-width: 720px; }
  .field label { display: block; font-weight: 600; font-size: 13px; margin-bottom: 4px; }
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
  .secret { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
  .secret .state { color: var(--muted); font-size: 12px; min-width: 50px; }
  .secret input { flex: 1; }
  .secfoot { display: flex; align-items: center; gap: 12px; padding: 10px 18px; border-top: 1px solid var(--line); }
  .saved { color: var(--accent); font-size: 13px; }
  .spacer { flex: 1; }
  code { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
</style>
