<script lang="ts">
  // The settings window: every field of sidecoach.toml, generated from the pydantic
  // schema the sidecar serves (descriptions, choices, ranges, and when a change applies),
  // the API keys into the OS credential store, the audio devices by name, and the
  // shell's own two switches (what closing the window does, start with the system).
  import { onMount } from "svelte";
  import { connect, get, post, put } from "$lib/api";
  import { inShell, setShellSettings, shellSettings, type ShellSettings } from "$lib/shell.svelte";
  import { voiceLabel } from "$lib/voices";

  type Prop = { type?: string; anyOf?: { type: string }[]; description?: string; default?: unknown; choices?: string[]; applies?: string; kind?: string; group?: string; advanced?: boolean; dev?: boolean; minimum?: number; maximum?: number; title?: string };
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
  let result = $state<{ ok: boolean; now?: string[]; next?: string[]; restart?: string[]; errors?: { loc: (string | number)[]; msg: string }[] } | null>(null);
  let section = $state("llm");

  const SECTIONS: [string, string][] = [
    ["llm", "AI Coach"], ["coach", "Coach"], ["voice", "Voice"], ["engine", "Engine"],
    ["capture", "Screen"], ["profile", "You"], ["gpu", "GPU"], ["app", "App"],
  ];
  // The AI Coach page in sections: which model, then what applies to a model served on
  // this machine, to a service, and to both. A field's group comes from the schema.
  const LLM_GROUPS: [string, string, string][] = [
    ["provider", "Provider", "Which model coaches: a service, a model hosted on this machine, or none at all."],
    ["local", "Local hosting", "A model served on this machine or the network (LM Studio, Ollama, vLLM): the local preset, or a custom one."],
    ["service", "Service providers", "OpenAI, DashScope, or another OpenAI-compatible service: the key, the effort, the prices for the session's cost."],
    ["general", "General", "What applies whichever model it is."],
  ];
  let listening = $state("");

  function fields(name: string, group?: string): [string, Prop][] {
    if (!data) return [];
    const ref = data.schema.properties[name]?.$ref;
    if (!ref) return [];
    const model = data.schema.$defs[ref.split("/").pop()!];
    return Object.entries(model?.properties ?? {}).filter(([, p]) => (advanced || !p.advanced) && !(inShell() && p.dev) && (!group || (p.group ?? "general") === group));
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
  const secretNames = $derived.by(() => {
    if (!data) return [] as string[];
    const names = ["OPENAI_API_KEY", "DASHSCOPE_API_KEY"];
    const custom = values.llm?.api_key_env;
    if (custom && !names.includes(custom)) names.push(custom);
    return names;
  });
  const appliesWord: Record<string, string> = { now: "applies at once", next: "applies at the next Start", restart: "needs a restart" };
  /** "voice.voice" -> "Voice"; "voice.tts_device" -> "TTS device"; "llm.provider" -> "Provider". */
  function fieldName(path: string): string {
    const key = path.split(".").pop() ?? path;
    const words = key.split("_").map((w) => (["tts", "stt", "llm", "gpu", "url", "api", "ptt"].includes(w) ? w.toUpperCase() : w));
    const text = words.join(" ");
    return text[0].toUpperCase() + text.slice(1);
  }
  /** The line under Save, as a sentence: what applied now, what waits for a Start or a restart. */
  function savedLine(r: NonNullable<typeof result>): string {
    const parts: string[] = [];
    if (r.now?.length) parts.push(`Applied now: ${r.now.map(fieldName).join(", ")}.`);
    if (r.next?.length) parts.push(`At the next Start: ${r.next.map(fieldName).join(", ")}.`);
    if (r.restart?.length) parts.push(`After a restart: ${r.restart.map(fieldName).join(", ")}.`);
    return parts.length ? "Saved. " + parts.join(" ") : "Saved. Nothing changed.";
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
    for (const name of secretNames) {
      if (clearSecrets[name]) sec[name] = null;
      else if (secrets[name]) sec[name] = secrets[name];
    }
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
        <label for="{section}-{key}">{key.replaceAll("_", " ")}</label>
      </div>
    {:else}
      <label for="{section}-{key}">{key.replaceAll("_", " ")}</label>
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
          {#each p.choices as c (c)}<option value={c} selected={(v == null ? "" : String(v)) === c}>{c === "" ? "(preset default)" : c}</option>{/each}
        </select>
      {:else if t === "integer" || t === "number"}
        <input id="{section}-{key}" type="number" value={v ?? ""} min={p.minimum} max={p.maximum} step={t === "integer" ? 1 : "any"} oninput={(e) => setValue(section, key, p, e.currentTarget.value)} />
      {:else if t === "object"}
        <input id="{section}-{key}" type="text" value={JSON.stringify(v ?? {})} onchange={(e) => { try { values[section][key] = JSON.parse(e.currentTarget.value || "{}"); } catch { /* keep typing */ } }} />
      {:else}
        <input id="{section}-{key}" type="text" value={v ?? ""} placeholder={isNullable(p) ? "(unset)" : ""} oninput={(e) => setValue(section, key, p, e.currentTarget.value)} />
      {/if}
    {/if}
    <p class="desc">{p.description}{#if p.applies} <span class="applies {p.applies}">· {appliesWord[p.applies] ?? p.applies}</span>{/if}</p>
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
        <div class="field row">
          <input id="autostart" type="checkbox" bind:checked={shell.autostart} />
          <label for="autostart">Start with the system <span class="muted">(not wired yet)</span></label>
        </div>
      {:else}
        <p class="muted">The shell's own settings show in the desktop app.</p>
      {/if}
      <div class="field">
        <label for="data_dir">Data folder</label>
        <input id="data_dir" type="text" value={values.data_dir} disabled={inShell()} oninput={(e) => (values.data_dir = e.currentTarget.value)} />
        <p class="desc">{data.schema.properties.data_dir?.description}{inShell() ? " The app decides where." : ""}</p>
      </div>
      <div class="field">
        <label for="tools_dir">Tools folder</label>
        <input id="tools_dir" type="text" value={values.tools_dir} oninput={(e) => (values.tools_dir = e.currentTarget.value)} />
        <p class="desc">{data.schema.properties.tools_dir?.description}</p>
      </div>
      <p class="muted">Settings file: <code>{data.path}</code></p>
    {:else}
      <h2>{SECTIONS.find(([id]) => id === section)?.[1]}</h2>
      {#if section === "llm"}
        {#each LLM_GROUPS as [group, title, blurb] (group)}
          {@const list = fields(section, group)}
          {#if list.length || group === "service"}
            <h3>{title}</h3>
            <p class="desc lead">{blurb}</p>
            {#if group === "service"}
              <div class="field">
                <span class="lbl">API keys</span>
                <p class="desc">{data.keyring ? "Kept in the operating system's credential store, never in the settings file." : "No credential store is available in this environment; keys come from the environment or .env."}</p>
                {#each secretNames as name (name)}
                  <div class="secret">
                    <code>{name}</code>
                    <span class="state">{data.secrets[name] ? "set" : "not set"}</span>
                    <input type="password" placeholder={data.secrets[name] ? "replace…" : "paste the key…"} value={secrets[name] ?? ""} oninput={(e) => (secrets[name] = e.currentTarget.value)} disabled={!data.keyring} />
                    {#if data.secrets[name]}<label class="ctl"><input type="checkbox" bind:checked={clearSecrets[name]} /> forget</label>{/if}
                  </div>
                {/each}
              </div>
            {/if}
            {#each list as [key, p] (key)}{@render field(section, key, p)}{/each}
          {/if}
        {/each}
      {:else}
        {#each fields(section) as [key, p] (key)}{@render field(section, key, p)}{/each}
      {/if}
    {/if}
  </div>

  <footer class="secfoot">
    {#if result?.ok}
      <span class="saved">{savedLine(result)}</span>
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
  h3 { margin: 18px 0 4px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
  .lead { margin: 0 0 10px; }
  .listen { flex: 0 0 auto; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 6px 12px; font: inherit; font-weight: 600; cursor: pointer; }
  .listen:disabled { opacity: 0.6; cursor: default; }
  .field { margin: 0 0 14px; max-width: 720px; }
  .field label, .field .lbl { display: block; font-weight: 600; font-size: 13px; margin-bottom: 4px; text-transform: capitalize; }
  .field .row { display: flex; align-items: center; gap: 8px; }
  .field .row label { margin: 0; }
  .field input[type="text"], .field input[type="number"], .field input[type="password"], .field select {
    width: 100%; box-sizing: border-box; background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; font: inherit;
  }
  .field input[type="number"] { max-width: 180px; }
  .field.error input, .field.error select { border-color: var(--error); }
  .desc { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
  .applies.now { color: var(--accent); }
  .applies.restart { color: var(--warn); }
  .err { color: var(--error); font-size: 12px; margin: 4px 0 0; }
  .muted { color: var(--muted); font-size: 12px; }
  .secret { display: flex; gap: 8px; align-items: center; margin: 6px 0; }
  .secret code { min-width: 150px; }
  .secret .state { color: var(--muted); font-size: 12px; min-width: 50px; }
  .secret input { flex: 1; }
  .secfoot { display: flex; align-items: center; gap: 12px; padding: 10px 18px; border-top: 1px solid var(--line); }
  .saved { color: var(--accent); font-size: 13px; }
  .spacer { flex: 1; }
  code { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
</style>
