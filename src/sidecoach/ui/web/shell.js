// The desktop shell's native menu bar and tray, built from the page with Tauri's JS API
// when the page runs inside the shell (window.__TAURI__ is injected there and nowhere
// else). In a browser this file does nothing: the header's own controls stay the way in.
//
// Phase 0 of docs/app-plan.md: the menus drive the same API and DOM the header does, so
// the shell adds a menu bar and takes nothing away. Loaded after app.js, it uses its
// globals (api, el, filters, applyFilters, fitBoard, prefs).

// Without devtools at hand, whatever goes wrong in here is reported to the sidecar
// (data/perf/shell.jsonl) as well as to the console.
function report(stage, error) {
  console.error("shell:", stage, error);
  post("/api/perf", { label: "shell", sample: { stage, error: String(error && error.stack || error) } }).catch(() => {});
}

(async function shell() {
  const tauri = window.__TAURI__;
  if (!tauri) return;
  if (!tauri.menu) { report("api", "window.__TAURI__ has no menu module: " + Object.keys(tauri).join(",")); return; }
  try {
    await build(tauri);
    post("/api/perf", { label: "shell", sample: { stage: "menu", ok: true } }).catch(() => {});
  } catch (e) {
    report("build", e);
  }
})();

async function build(tauri) {
  const { Menu, Submenu, MenuItem, CheckMenuItem, PredefinedMenuItem } = tauri.menu;
  const win = tauri.window && tauri.window.getCurrentWindow ? tauri.window.getCurrentWindow() : null;

  // --- what the View items switch ---------------------------------------------------------
  const panes = {
    board: { text: "Board", node: () => el("left"), key: "sidecoach.view.board" },
    moves: { text: "Move list", nodes: () => [el("moves").parentNode, el("moves").parentNode.previousElementSibling, el("splitMoves")], key: "sidecoach.view.moves" },
    strip: { text: "Status strip", node: () => el("health"), key: "sidecoach.view.strip" },
  };
  const shown = (p) => localStorage.getItem(p.key) !== "0";
  const applyPane = (p) => {
    const nodes = p.nodes ? p.nodes() : [p.node()];
    for (const n of nodes) if (n) n.hidden = !shown(p);
    if (p === panes.board) document.documentElement.classList.toggle("noboard", !shown(p));
    fitBoard();
  };
  for (const p of Object.values(panes)) applyPane(p);
  const terminalShown = () => !document.querySelector("footer").classList.contains("folded");

  const item = (text, action, opts) => MenuItem.new({ text, action, ...(opts || {}) });
  const check = (text, checked, action, opts) => CheckMenuItem.new({ text, checked, action, ...(opts || {}) });

  // --- File --------------------------------------------------------------------------------
  const startItem = await item("Start watching", () => (running ? api.stop() : api.start()), { id: "start" });
  const file = await Submenu.new({
    text: "File",
    items: [
      startItem,
      await PredefinedMenuItem.new({ item: "Separator" }),
      await item("Dump the last frames and the log", () => post("/api/dump")),
      await PredefinedMenuItem.new({ item: "Separator" }),
      await PredefinedMenuItem.new({ item: "Quit", text: "Quit Sidecoach" }),
    ],
  });

  // --- Game --------------------------------------------------------------------------------
  const game = await Submenu.new({
    text: "Game",
    items: [
      await item("I resigned / lost on time", () => api.end("resigned")),
      await item("Opponent resigned / flagged", () => api.end("won")),
      await item("Draw agreed", () => api.end("draw")),
      await PredefinedMenuItem.new({ item: "Separator" }),
      await item("Forget this game", () => api.end("forget")),
      await PredefinedMenuItem.new({ item: "Separator" }),
      await item("Re-read the screen", () => api.analyze()),
      await item("Flip the board", () => api.flip()),
    ],
  });

  // --- View --------------------------------------------------------------------------------
  const viewItems = [];
  for (const [id, p] of Object.entries(panes)) {
    viewItems.push(await check(p.text, shown(p), async (e) => {
      localStorage.setItem(p.key, shown(p) ? "0" : "1");
      applyPane(p);
    }, { id: "view-" + id }));
  }
  const termItem = await check("Terminal", terminalShown(), () => el("fold").click(), { id: "view-terminal" });
  const topItem = await check("Always on top", false, async () => {
    if (!win) return;
    const on = await win.isAlwaysOnTop();
    await win.setAlwaysOnTop(!on);
  }, { id: "view-top" });
  const view = await Submenu.new({
    text: "View",
    items: [
      ...viewItems,
      termItem,
      await PredefinedMenuItem.new({ item: "Separator" }),
      topItem,
      await PredefinedMenuItem.new({ item: "Separator" }),
      await item("Reset layout", () => {
        for (const k of Object.keys(localStorage)) if (k.startsWith("sidecoach.")) localStorage.removeItem(k);
        location.reload();
      }),
    ],
  });

  // --- Voice -------------------------------------------------------------------------------
  const muteItem = await check("Mute", el("mute").getAttribute("aria-pressed") === "true", () => el("mute").click(), { id: "voice-mute" });
  const listen = {};
  for (const [mode, text] of [["off", "Not listening"], ["ptt", "Push to talk"], ["vad", "Voice activity"]]) {
    listen[mode] = await check(text, false, () => post("/api/listen", { mode }), { id: "listen-" + mode });
  }
  const voice = await Submenu.new({
    text: "Voice",
    items: [muteItem, await PredefinedMenuItem.new({ item: "Separator" }), listen.off, listen.ptt, listen.vad],
  });

  // --- Data --------------------------------------------------------------------------------
  const data = await Submenu.new({
    text: "Data",
    items: [
      await item("Clear caches", () => api.clear_caches()),
      await PredefinedMenuItem.new({ item: "Separator" }),
      await item("Reset profile…", async () => {
        if (window.confirm("Reset the profile? Every game, verdict, puzzle attempt and insight the player model holds is deleted. This cannot be undone.")) {
          await api.reset_profile();
          el("coach").replaceChildren();
        }
      }),
    ],
  });

  // --- Debug -------------------------------------------------------------------------------
  const verboseItem = await check("Verbose terminal", filters.verbose, () => { el("verbose").click(); }, { id: "debug-verbose" });
  const debug = await Submenu.new({
    text: "Debug",
    items: [
      verboseItem,
      await item("Open the panel in the browser", () => tauri.opener && tauri.opener.openUrl ? tauri.opener.openUrl(location.origin + "/") : window.open(location.origin + "/")),
    ],
  });

  // --- Help --------------------------------------------------------------------------------
  const help = await Submenu.new({
    text: "Help",
    items: [
      await item("Sidecoach on GitHub", () => tauri.opener && tauri.opener.openUrl && tauri.opener.openUrl("https://github.com/ductai199x/sidecoach")),
      await PredefinedMenuItem.new({ item: { About: { name: "Sidecoach", comments: "Vision-first chess coaching companion" } }, text: "About Sidecoach" }),
    ],
  });

  const menu = await Menu.new({ items: [file, game, view, voice, data, debug, help] });
  await menu.setAsAppMenu();

  // --- keep the check items honest against the page's state ------------------------------
  setInterval(async () => {
    try {
      await startItem.setText(running ? "Stop watching" : "Start watching");
      await termItem.setChecked(terminalShown());
      await muteItem.setChecked(el("mute").getAttribute("aria-pressed") === "true");
      await verboseItem.setChecked(!!filters.verbose);
      for (const [mode, it] of Object.entries(listen)) await it.setChecked(listenMode === mode);
      if (win) await topItem.setChecked(await win.isAlwaysOnTop());
    } catch (e) { /* the menu may be gone while the window closes */ }
  }, 1000);

  // --- tray --------------------------------------------------------------------------------
  if (tauri.tray && tauri.tray.TrayIcon) {
    try {
      const trayMenu = await Menu.new({
        items: [
          await item("Show Sidecoach", async () => { if (win) { await win.show(); await win.setFocus(); } }),
          await PredefinedMenuItem.new({ item: "Quit", text: "Quit" }),
        ],
      });
      await tauri.tray.TrayIcon.new({ menu: trayMenu, tooltip: "Sidecoach", icon: await tauri.app.defaultWindowIcon(), showMenuOnLeftClick: true });
    } catch (e) {
      report("tray", e);
    }
  }
}
