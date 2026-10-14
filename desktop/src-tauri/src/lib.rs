//! The Sidecoach desktop shell: a window and a tray around the Python
//! pipeline, which runs as a sidecar behind its own HTTP API (docs/app.md).
//!
//! The shell knows nothing about chess. It spawns the sidecar (`sidecoach panel --port 0
//! --announce --parent-pid <us> --no-persist --no-record`), reads the one JSON line the
//! sidecar prints with its port, waits for `/api/health`, hands the port and a bearer
//! token to the page, and kills the sidecar on exit; the sidecar also exits by itself when
//! this process is gone. The window has no menu bar: the page's drawer is the way in, on
//! the desktop as on the phone (Tai, 2026-09-24), and the tray is built from the page with
//! the JS API (desktop/src/lib/shell.svelte.ts). Closing the main window hides it to the
//! tray or quits, as the shell's own setting says (Tai, 2026-09-21); the puzzle and
//! settings pages open in windows of their own.
//!
//! Development (`tauri dev`, or a debug build): the sidecar is `uv run --no-sync sidecoach
//! panel ...` in the repository root, found by walking up to `pyproject.toml`; extra
//! arguments come from `SIDECOACH_SIDECAR_ARGS` (a replay, a config). A release build runs
//! the embedded interpreter from the bundle's resources.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, Emitter, Manager, RunEvent, State, WebviewUrl, WebviewWindowBuilder, WindowEvent,
};

mod places;

/// How long the shell waits for the sidecar's port line and for its health route.
const START_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Default)]
struct Sidecar {
    child: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
    /// Why the sidecar is not serving, when it is not: the splash screen shows it.
    error: Mutex<Option<String>>,
    token: String,
}

/// What the splash screen polls until the sidecar serves or fails.
#[derive(Clone, Serialize)]
struct SidecarStatus {
    port: Option<u16>,
    error: Option<String>,
    /// The sidecar's process has ended: the page's "Sidecoach stopped" rests on this alone,
    /// never on polls that failed (a live sidecar answering errors is not stopped).
    exited: bool,
}

/// The shell's own settings, kept apart from the sidecar's TOML because the shell needs
/// them before the sidecar is up: what closing the window does, and start with the OS.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct ShellSettings {
    close: String, // "quit" | "tray"
    autostart: bool,
}

impl Default for ShellSettings {
    fn default() -> Self {
        Self {
            // Closing the window quits (Tai, 2026-09-21); the tray is opted into.
            close: "quit".into(),
            autostart: false,
        }
    }
}

#[derive(Default)]
struct Shell {
    settings: Mutex<ShellSettings>,
}

fn settings_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("shell.json"))
}

fn load_shell_settings(app: &AppHandle) -> ShellSettings {
    settings_path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_shell_settings(app: &AppHandle, s: &ShellSettings) {
    if let Some(p) = settings_path(app) {
        if let Some(dir) = p.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(p, serde_json::to_string_pretty(s).unwrap_or_default());
    }
}

/// The repository root in development: the first ancestor of the working directory or
/// the executable with a `pyproject.toml` and the package.
fn repo_root() -> Option<PathBuf> {
    let mut starts = vec![std::env::current_dir().ok()];
    starts.push(
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(PathBuf::from)),
    );
    for start in starts.into_iter().flatten() {
        let mut dir = Some(start.as_path());
        while let Some(d) = dir {
            if d.join("pyproject.toml").exists() && d.join("src").join("sidecoach").exists() {
                return Some(d.to_path_buf());
            }
            dir = d.parent();
        }
    }
    None
}

/// The per-user data directory the sidecar is pointed at in the product; in development
/// the repository's `data/` (the CLI's default) is kept, so recordings and the profile
/// stay where the tools expect them.
fn data_dir(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("SIDECOACH_DATA_DIR") {
        return Some(PathBuf::from(dir));
    }
    if embedded_python(app).is_some() {
        return app.path().app_local_data_dir().ok().map(|d| d.join("data"));
    }
    None
}

/// The embedded interpreter: the bundle's `resources/python`, or `SIDECOACH_PYTHON` (a
/// build of it, to run the app's mode out of a development checkout).
fn embedded_python(app: &AppHandle) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SIDECOACH_PYTHON") {
        let p = PathBuf::from(p);
        return if p.exists() { Some(p) } else { None };
    }
    app.path()
        .resource_dir()
        .ok()
        .map(|r| {
            r.join("python").join(if cfg!(windows) {
                "python.exe"
            } else {
                "bin/python3"
            })
        })
        .filter(|p| p.exists())
}

/// The command that runs the sidecar: the embedded interpreter when the bundle carries
/// one, else `uv run` in the repository (development).
fn sidecar_command(app: &AppHandle, token: &str) -> Result<Command, String> {
    let mut cmd = if let Some(python) = embedded_python(app) {
        let mut c = Command::new(&python);
        // In the app's own resources, whatever folder the shell was started from: the sidecar
        // reads a `.env` in its working folder, and Linux starts apps in the home folder, where
        // one may hold anything (the Linux audit, 2026-10-13). Windows' was the install folder.
        if let Ok(resources) = app.path().resource_dir() {
            c.current_dir(resources);
        }
        // Isolated: the embedded interpreter ignores the environment's PYTHON* variables
        // and the user's site-packages. An AppImage's launcher exports PYTHONHOME and
        // PYTHONPATH for any Python inside it, which sent this one looking for its
        // standard library in the wrong place (Phase 2, 2026-09-21).
        c.arg("-I");
        // The bytecode cache goes under the data directory, not into the install folder:
        // the uninstaller removes only what the installer wrote, and an AppImage is
        // read-only anyway. As an option, since -I ignores PYTHONPYCACHEPREFIX too.
        if let Some(dir) = data_dir(app) {
            c.arg(format!(
                "-Xpycache_prefix={}",
                dir.join("pycache").display()
            ));
        }
        c.args(["-m", "sidecoach", "panel"]);
        c.env_remove("PYTHONHOME").env_remove("PYTHONPATH");
        #[cfg(target_os = "linux")]
        without_appimage_env(&mut c);
        c
    } else {
        let root = repo_root()
            .ok_or("no embedded Python and no repository root (pyproject.toml) found")?;
        let mut c = Command::new("uv");
        c.args(["run", "--no-sync", "sidecoach", "panel"]);
        c.current_dir(root);
        c
    };
    cmd.args([
        "--port",
        "0",
        "--announce",
        "--parent-pid",
        &std::process::id().to_string(),
        "--no-persist",
        "--no-record",
    ]);
    if !token.is_empty() {
        // By the environment, not the command line any local process can read.
        cmd.env("SIDECOACH_PANEL_TOKEN", token);
    }
    if let Some(dir) = data_dir(app) {
        let _ = std::fs::create_dir_all(&dir);
        cmd.env("SIDECOACH_DATA_DIR", dir);
        // The settings file beside the shell's own, in the config dir; keys never in it. One
        // named already is kept, as the data directory is: a run of the installed app on a
        // copy of someone's data must not write their own settings.
        if std::env::var_os("SIDECOACH_CONFIG").is_none() {
            if let Ok(config) = app.path().app_config_dir() {
                let _ = std::fs::create_dir_all(&config);
                cmd.env("SIDECOACH_CONFIG", config.join("sidecoach.toml"));
            }
        }
    }
    if let Ok(extra) = std::env::var("SIDECOACH_SIDECAR_ARGS") {
        cmd.args(extra.split_whitespace());
    }
    Ok(cmd)
}

/// One HTTP GET to the sidecar's health route; true on a 200.
fn healthy(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}");
    let Ok(mut stream) =
        TcpStream::connect_timeout(&addr.parse().expect("address"), Duration::from_millis(500))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream
        .write_all(b"GET /api/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut head = [0u8; 64];
    let n = stream.read(&mut head).unwrap_or(0);
    let text = String::from_utf8_lossy(&head[..n]);
    text.starts_with("HTTP/1.") && text.contains(" 200")
}

/// Spawn the sidecar and return its port once it serves. The child is kept to be killed
/// at exit; a sidecar that only announced another instance's port exits on its own.
fn start_sidecar(app: &AppHandle, state: &Sidecar) -> Result<u16, String> {
    let mut cmd = sidecar_command(app, &state.token)?;
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    // The sidecar's stderr goes to a log file under the data directory when there is one
    // (a release build has no console), else to ours.
    let stderr = data_dir(app)
        .and_then(|d| std::fs::create_dir_all(&d).ok().map(|_| d))
        .and_then(|d| {
            keep_previous_log(&d);
            std::fs::File::create(d.join("sidecar.log")).ok()
        })
        .map(Stdio::from)
        .unwrap_or_else(Stdio::inherit);
    cmd.stdout(Stdio::piped())
        .stderr(stderr)
        .stdin(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("cannot start the sidecar: {e}"))?;
    let stdout = child.stdout.take().ok_or("no stdout from the sidecar")?;
    let started = Instant::now();
    // A thread drains stdout for the sidecar's whole life (a full pipe would block it,
    // a closed one would break its prints) and hands over the one line that matters.
    let (tx, rx) = std::sync::mpsc::channel::<u16>();
    std::thread::spawn(move || {
        let mut announced = false;
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if !announced && line.starts_with('{') {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let Some(p) = v.get("port").and_then(|p| p.as_u64()) {
                        announced = true;
                        let _ = tx.send(p as u16);
                        continue;
                    }
                }
            }
            eprintln!("[sidecar] {line}");
        }
    });
    let port = match rx.recv_timeout(START_TIMEOUT) {
        Ok(port) => port,
        Err(_) => {
            let _ = child.kill();
            return Err("the sidecar did not announce a port in time".into());
        }
    };
    *state.child.lock().unwrap() = Some(child);
    while !healthy(port) {
        if started.elapsed() > START_TIMEOUT {
            return Err(format!(
                "the sidecar on port {port} never answered /api/health"
            ));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    *state.port.lock().unwrap() = Some(port);
    Ok(port)
}

/// How long a quit waits for the sidecar to close by itself (the session's end: the last
/// game's review and profile, Stockfish) before it is killed.
const STOP_TIMEOUT: Duration = Duration::from_secs(25);

/// Ask the sidecar to end itself: one POST to its shutdown route; true on a 200.
fn ask_shutdown(port: u16, token: &str) -> bool {
    let addr = format!("127.0.0.1:{port}");
    let Ok(mut stream) =
        TcpStream::connect_timeout(&addr.parse().expect("address"), Duration::from_millis(500))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let request = format!(
        "POST /api/shutdown HTTP/1.0\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\nContent-Length: 0\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut head = [0u8; 64];
    let n = stream.read(&mut head).unwrap_or(0);
    let text = String::from_utf8_lossy(&head[..n]);
    text.starts_with("HTTP/1.") && text.contains(" 200")
}

fn stop_sidecar(app: &AppHandle, state: &Sidecar) {
    let port = *state.port.lock().unwrap();
    stop_sidecar_on(app, state, port);
}

/// `stop_sidecar` for a sidecar serving on `port` (None: it never served, it is killed).
fn stop_sidecar_on(app: &AppHandle, state: &Sidecar, port: Option<u16>) {
    if let Some(mut child) = state.child.lock().unwrap().take() {
        // An orderly end first, so the sidecar's own close runs; a bounded wait; then
        // the kill for a sidecar that hangs or never served.
        let asked = port.is_some_and(|p| ask_shutdown(p, &state.token));
        let started = Instant::now();
        let mut exited = false;
        while asked && started.elapsed() < STOP_TIMEOUT {
            if matches!(child.try_wait(), Ok(Some(_))) {
                exited = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if !exited {
            let _ = child.kill();
        }
        let _ = child.wait();
        // A kill skips the sidecar's own cleanup; its lock file names a dead process.
        // (The sidecar removes its own lock when it closes in order.)
        if let Some(dir) = data_dir(app).or_else(|| repo_root().map(|r| r.join("data"))) {
            let _ = std::fs::remove_file(dir.join("panel.lock"));
        }
    }
}

/// Stops the sidecar when dropped. Kept in the app's resource table: the updater's install
/// (Windows) exits the process without a `RunEvent::Exit`, after `cleanup_before_exit`,
/// which clears that table -- so the sidecar no longer holds the files the installer is
/// about to replace (setup finding 10).
struct SidecarGuard(AppHandle);

impl tauri::Resource for SidecarGuard {}

impl Drop for SidecarGuard {
    fn drop(&mut self) {
        stop_sidecar(&self.0, &self.0.state::<Sidecar>());
    }
}

/// The sidecar's log of the previous launch kept as `sidecar.1.log` before a new one
/// starts: truncating it at every launch lost the trace of the crash that ended the last
/// (setup finding 22).
fn keep_previous_log(dir: &std::path::Path) {
    let log = dir.join("sidecar.log");
    if log.exists() {
        let _ = std::fs::rename(&log, dir.join("sidecar.1.log"));
    }
}

fn random_token() -> String {
    // The OS's CSPRNG: a bearer guessable from the time and pid is none.
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes).expect("the OS random source");
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---- what the page can ask -----------------------------------------------------------------

#[tauri::command]
fn sidecar_port(state: State<'_, Sidecar>) -> Option<u16> {
    *state.port.lock().unwrap()
}

#[tauri::command]
fn sidecar_status(state: State<'_, Sidecar>) -> SidecarStatus {
    SidecarStatus {
        port: *state.port.lock().unwrap(),
        error: state.error.lock().unwrap().clone(),
        exited: child_exited(&mut state.child.lock().unwrap()),
    }
}

/// Whether the child the shell holds has ended. None held (not started yet, or a Restart
/// taking the old one away) is not an exit.
fn child_exited(child: &mut Option<Child>) -> bool {
    child
        .as_mut()
        .is_some_and(|c| matches!(c.try_wait(), Ok(Some(_))))
}

/// The page found the sidecar gone (or it never started) and the player pressed Restart:
/// what is left of the old one goes, and a new one starts. Returns at once; the page waits
/// through `sidecar_status` for the new port or the error.
#[tauri::command]
fn restart_sidecar(app: AppHandle, state: State<'_, Sidecar>) {
    // Cleared before the answer, so the page's wait sees only the new sidecar. The old port
    // is kept for the old sidecar's close: one still serving (the page may have failed its
    // polls for another reason) is asked to close in order, as at a quit, not killed with
    // the session's end unwritten; one that is gone is killed after the ask fails.
    let old = state.port.lock().unwrap().take();
    *state.error.lock().unwrap() = None;
    std::thread::spawn(move || {
        let state = app.state::<Sidecar>();
        stop_sidecar_on(&app, &state, old);
        if let Err(err) = start_sidecar(&app, &state) {
            eprintln!("sidecar: {err}");
            *state.error.lock().unwrap() = Some(err);
        }
    });
}

/// The splash screen has shown the warm-up through (or was dismissed): the main window
/// comes up and the splash goes away. Harmless when there is no splash.
#[tauri::command]
fn splash_done(app: AppHandle) {
    show_main(&app);
    if let Some(splash) = app.get_webview_window("splash") {
        let _ = splash.close();
    }
}

/// The splash is about to fade: the main window comes up behind it, the splash kept in front,
/// so the card fades away over the app rather than to a blank screen (Tai, 2026-10-03: "smooth
/// fade out to the app"). `splash_done` then closes the splash and gives the main window focus.
#[tauri::command]
fn splash_reveal(app: AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        places::shown(&window);
    }
    if let Some(splash) = app.get_webview_window("splash") {
        let _ = splash.set_focus();
    }
}

fn show_main(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        places::shown(&window);
        let _ = window.set_focus();
    }
}

/// The splash screen: a small frameless window over the desktop while the sidecar
/// starts and warms up (docs/app.md, "everything loads when the app opens"). The
/// main window stays hidden until the splash says it is done. `SIDECOACH_NO_SPLASH`
/// skips it (the smoke test, development), showing the main window at once.
fn open_splash(app: &AppHandle) -> bool {
    if std::env::var("SIDECOACH_NO_SPLASH").is_ok() {
        return false;
    }
    let built = WebviewWindowBuilder::new(app, "splash", WebviewUrl::App("splash".into()))
        .title("Sidecoach")
        .inner_size(720.0, 480.0)
        .resizable(false)
        .decorations(false)
        // Transparent, the card drawing its own rounded corners: it fades away over the main
        // window shown behind it (`splash_reveal`). A window's own shadow would stay behind
        // as a dark frame while the card faded, so there is none.
        .transparent(true)
        .shadow(false)
        // An ordinary window: never on top of everything, and in the taskbar and Alt+Tab so
        // it can be brought back once something else is in front (Tai, 2026-09-24: "it's
        // basically on top of everything else and I can't bring something else in front").
        .always_on_top(false)
        .skip_taskbar(false)
        .center()
        .build();
    match built {
        Ok(_) => true,
        Err(err) => {
            eprintln!("splash: {err}");
            false
        }
    }
}

#[tauri::command]
fn sidecar_token(state: State<'_, Sidecar>) -> String {
    state.token.clone()
}

#[tauri::command]
fn data_dir_command(app: AppHandle) -> String {
    data_dir(&app)
        .or_else(|| repo_root().map(|r| r.join("data")))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// The data folder opened in the system's file manager (the drawer's "Open data folder"). The
/// page asked for a command named `data_dir` that was never registered, and the opener's
/// default permission does not open paths: the button did nothing (the Linux audit,
/// 2026-10-13). Opened here, by the shell, it needs no permission of the page's.
#[tauri::command]
fn open_data_folder(app: AppHandle) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let path = data_dir_command(app.clone());
    if path.is_empty() {
        return Err("no data folder".into());
    }
    let _ = std::fs::create_dir_all(&path);
    app.opener()
        .open_path(path, None::<&str>)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn shell_settings(shell: State<'_, Shell>) -> ShellSettings {
    shell.settings.lock().unwrap().clone()
}

#[derive(Deserialize)]
struct ShellSettingsPatch {
    close: Option<String>,
    autostart: Option<bool>,
}

#[tauri::command]
fn set_shell_settings(
    app: AppHandle,
    shell: State<'_, Shell>,
    settings: ShellSettingsPatch,
) -> ShellSettings {
    let mut current = shell.settings.lock().unwrap();
    if let Some(close) = settings.close {
        if close == "tray" || close == "quit" {
            current.close = close;
        }
    }
    if let Some(autostart) = settings.autostart {
        current.autostart = autostart;
    }
    save_shell_settings(&app, &current);
    current.clone()
}

/// Errors from before a page's own code runs (a page that never boots) reach the terminal
/// through the sidecar; the page installs the same handlers once connected.
fn page_report(app: &AppHandle) -> String {
    let state = app.state::<Sidecar>();
    let port = *state.port.lock().unwrap();
    match (port, state.token.as_str()) {
        (Some(port), token) => format!(
            r#"(() => {{
  const send = (text) => {{ try {{ fetch("http://127.0.0.1:{port}/api/client_log", {{ method: "POST", keepalive: true,
    headers: {{ "Content-Type": "application/json", "Authorization": "Bearer {token}" }},
    body: JSON.stringify({{ text: String(text).slice(0, 2000), page: location.pathname }}) }}).catch(() => {{}}); }} catch (_) {{}} }};
  window.addEventListener("error", (e) => send((e.message || e) + " @ " + (e.filename || "?") + ":" + (e.lineno || 0)));
  window.addEventListener("unhandledrejection", (e) => send("unhandled rejection: " + (e.reason && (e.reason.stack || e.reason.message) || e.reason)));
  if ({trace}) {{
    send("boot " + location.href);
    document.addEventListener("DOMContentLoaded", () => send("dom " + document.documentElement.outerHTML.length + " chars: " + document.documentElement.outerHTML.slice(0, 400).replace(/\s+/g, " ")));
  }}
}})();"#,
            trace = std::env::var("SIDECOACH_PAGE_TRACE").is_ok(),
        ),
        _ => String::new(),
    }
}

/// The pointer passes through `window`. On Linux a window built hidden has no GDK window until
/// it is realized, and tao unwraps it: turning the captions on brought the whole app down
/// ("called `Option::unwrap()` on a `None` value", event_loop.rs:457 -- Ubuntu 24.04,
/// 2026-10-13). Realized first there, on GTK's thread, it never shows for that.
fn click_through(window: &tauri::WebviewWindow) {
    #[cfg(target_os = "linux")]
    {
        let w = window.clone();
        let _ = window.run_on_main_thread(move || {
            if let Ok(gtk_window) = w.gtk_window() {
                use gtk::prelude::WidgetExt;
                gtk_window.realize();
            }
            let _ = w.set_ignore_cursor_events(true);
        });
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = window.set_ignore_cursor_events(true);
    }
}

/// The captions' two windows, built hidden if they are not there yet: the box with the
/// coach's words (`captions`) and the cross a box is dropped on to turn them off
/// (`captions-cross`). Over every other window, without a frame or a taskbar button, and
/// never focused: a click on the box must not take the keyboard from the game (Tai,
/// 2026-09-29: "really useful in fast games"). Each page shows and places its own window.
#[tauri::command]
async fn open_captions(app: AppHandle) -> Result<(), String> {
    let report = page_report(&app);
    for (label, route, width, height) in [
        ("w-captions", "captions", 560.0, 140.0),
        ("w-captions-cross", "captions-cross", 96.0, 96.0),
    ] {
        if app.get_webview_window(label).is_some() {
            continue;
        }
        let window = WebviewWindowBuilder::new(&app, label, WebviewUrl::App(route.into()))
            .initialization_script(&report)
            .title("Sidecoach captions")
            .inner_size(width, height)
            .transparent(true)
            .decorations(false)
            .shadow(false)
            .resizable(false)
            .always_on_top(true)
            .skip_taskbar(true)
            .focused(false)
            .focusable(false)
            .visible(false)
            .build()
            .map_err(|e| e.to_string())?;
        if label == "w-captions-cross" {
            // Shown under a box being carried: the pointer passes through it to the box.
            click_through(&window);
        }
    }
    Ok(())
}

/// A page of the app in a window of its own: `settings`, `setup`, `games`, `puzzle/<id>` or
/// `review/<key>`. A second request for the same route brings the window it has to the front.
///
/// Async on purpose: a synchronous command runs on the main thread, and on Windows the
/// webview's creation completes through that thread's message loop -- built from a sync
/// command the window came up with a webview that never got its bounds, white and empty.
#[tauri::command]
async fn open_window(app: AppHandle, route: String) -> Result<(), String> {
    let route = route.trim_start_matches('/').to_string();
    // A place in the page (`review/<key>#ply-12`, from Stats) is not another window: the page's
    // own window is named without it, and one already open is sent there.
    let (page, place) = match route.split_once('#') {
        Some((page, place)) => (page.to_string(), Some(place.to_string())),
        None => (route.clone(), None),
    };
    let label: String = format!(
        "w-{}",
        page.chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>()
    );
    if let Some(w) = app.get_webview_window(&label) {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
        if let Some(place) = place {
            // Emptied first, so the same place asked for again is a change the page hears.
            let _ = w.eval(format!("location.hash = ''; location.hash = {:?}", format!("#{place}")));
        }
        return Ok(());
    }
    // The window's title and size by its page, whatever place in it was asked for.
    let (title, width, height) = if page.starts_with("puzzle/") {
        (
            format!("Sidecoach · {}", page.replace("puzzle/", "puzzle ")),
            1180.0,
            820.0,
        )
    } else if page.starts_with("review/") {
        ("Sidecoach · Game Review".to_string(), 1320.0, 880.0)
    } else if page == "stats" {
        ("Sidecoach · Stats".to_string(), 1180.0, 860.0)
    } else if page == "games" {
        ("Sidecoach · Games".to_string(), 1040.0, 800.0)
    } else if page == "settings" {
        ("Sidecoach · Settings".to_string(), 900.0, 760.0)
    } else {
        ("Sidecoach".to_string(), 900.0, 700.0)
    };
    let report = page_report(&app);
    // SIDECOACH_PAGE_TRACE: every asset the window asks the shell for, and what it got, in
    // <data dir>/shell-trace.log -- the way to see why a window came up blank.
    let trace = std::env::var("SIDECOACH_PAGE_TRACE").is_ok();
    let trace_file = data_dir(&app)
        .or_else(|| repo_root().map(|r| r.join("data")))
        .map(|d| d.join("shell-trace.log"));
    let mut builder =
        WebviewWindowBuilder::new(&app, &label, WebviewUrl::App(route.clone().into()))
            .initialization_script(&report)
            .title(title);
    if trace {
        if let Some(file) = trace_file {
            // What the resolver itself says for this route, before the webview asks.
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&file)
            {
                let found = app
                    .asset_resolver()
                    .get(route.clone())
                    .map(|a| a.bytes().len());
                let _ = writeln!(
                    f,
                    "resolver {route:?} -> {found:?}; url {:?}",
                    app.config().build.frontend_dist
                );
            }
            builder = builder.on_web_resource_request(move |request, response| {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&file)
                {
                    let _ = writeln!(
                        f,
                        "{} -> {} {} bytes",
                        request.uri(),
                        response.status(),
                        response.body().len()
                    );
                }
            });
        }
    }
    // Built hidden, put back where a window of its kind was last, then shown.
    let window = builder
        .inner_size(width, height)
        .min_inner_size(600.0, 420.0)
        .visible(false)
        .build()
        .map_err(|e| e.to_string())?;
    places::restore(&window);
    let _ = window.show();
    places::shown(&window);
    let _ = window.set_focus();
    if trace {
        // Where the webview says it is, a few seconds in.
        let file = data_dir(&app)
            .or_else(|| repo_root().map(|r| r.join("data")))
            .map(|d| d.join("shell-trace.log"));
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(4));
            if let Some(file) = file {
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&file)
                {
                    let _ = writeln!(
                        f,
                        "window url after 4 s: {:?}; eval: {:?}",
                        window.url(),
                        window.eval("void 0")
                    );
                }
            }
        });
    }
    Ok(())
}

/// WebKitGTK draws through DMA-BUF by default, and where the graphics driver cannot (a virtual
/// machine's, NVIDIA's own) every window stayed blank, on Wayland and on X11 alike (Ubuntu
/// 24.04, 2026-10-13). Shared memory draws everywhere; one set by the user is theirs.
///
/// The windows go through XWayland on a Wayland desktop, as the AppImage's launcher already
/// sends them (GDK_BACKEND=x11): a Wayland window may not place itself, stay on top, or say
/// where it is, so the captions box could not be put or dragged, Keep on top did nothing and
/// no window came back where it was left (the Linux audit, 2026-10-13). Wayland only where
/// there is no XWayland.
#[cfg(target_os = "linux")]
fn webkit_defaults() {
    // Before any thread exists: GTK and WebKit read them as they start.
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    if std::env::var_os("GDK_BACKEND").is_none() {
        std::env::set_var("GDK_BACKEND", "x11,wayland");
    }
}

/// The variables an AppImage's launcher points into its own folder (APPDIR): the library path,
/// GStreamer's plugins, GTK's modules, the data dirs. They are for the shell's GTK and WebKit;
/// the sidecar is a Python of its own, and with them it loaded the AppImage's GStreamer, which
/// has no PipeWire plugin, for the Wayland screen. Each one is given back without the AppImage's
/// entries, or removed when nothing else was in it.
#[cfg(target_os = "linux")]
fn without_appimage_env(cmd: &mut Command) {
    let Some(appdir) = std::env::var_os("APPDIR") else {
        return;
    };
    let appdir = appdir.to_string_lossy().trim_end_matches('/').to_string();
    if appdir.is_empty() {
        return;
    }
    for (name, value) in std::env::vars_os() {
        let value = value.to_string_lossy();
        if !value.contains(&appdir) {
            continue;
        }
        let kept: Vec<&str> = value
            .split(':')
            .filter(|part| !part.is_empty() && !part.starts_with(&appdir))
            .collect();
        if kept.is_empty() {
            cmd.env_remove(&name);
        } else {
            cmd.env(&name, kept.join(":"));
        }
    }
    // Forced by the AppImage for its GTK; the sidecar draws no window.
    cmd.env_remove("GDK_BACKEND");
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(target_os = "linux")]
    webkit_defaults();
    let sidecar = Sidecar {
        token: random_token(),
        ..Default::default()
    };
    let app = tauri::Builder::default()
        // First, so a second launch never gets as far as a second window: the installer's
        // run-if-installed, a second click on the shortcut, or a reopen while the panel sits
        // in the tray all show the running window instead.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(splash) = app.get_webview_window("splash") {
                let _ = splash.set_focus();
            } else {
                show_main(app);
            }
        }))
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .manage(sidecar)
        .manage(Shell::default())
        .manage(places::Places::default())
        .invoke_handler(tauri::generate_handler![
            sidecar_port,
            sidecar_status,
            restart_sidecar,
            splash_done,
            splash_reveal,
            sidecar_token,
            data_dir_command,
            open_data_folder,
            shell_settings,
            set_shell_settings,
            open_window,
            open_captions
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            *handle.state::<Shell>().settings.lock().unwrap() = load_shell_settings(&handle);
            // The main window comes up where it was closed (still hidden behind the splash).
            places::load(&handle);
            if let Some(main) = handle.get_webview_window("main") {
                places::restore(&main);
            }
            let splash = open_splash(&handle);
            if !splash {
                show_main(&handle);
            }
            app.resources_table().add(SidecarGuard(handle.clone()));
            std::thread::spawn(move || {
                let state = handle.state::<Sidecar>();
                match start_sidecar(&handle, &state) {
                    Ok(port) => {
                        // Development (the panel still served by the sidecar) may ask for
                        // the window to be pointed at it; the product keeps its own page.
                        if let Ok(query) = std::env::var("SIDECOACH_PAGE_QUERY") {
                            if let Some(window) = handle.get_webview_window("main") {
                                if let Ok(parsed) =
                                    format!("http://127.0.0.1:{port}/{query}").parse()
                                {
                                    let _ = window.navigate(parsed);
                                }
                            }
                        }
                        // Development: the bundled page with its meter on (`?perf=<label>`,
                        // samples to <data>/perf/<label>.jsonl), for a soak of the product path.
                        if let Ok(label) = std::env::var("SIDECOACH_PAGE_PERF") {
                            if let Some(window) = handle.get_webview_window("main") {
                                if let Ok(mut url) = window.url() {
                                    url.set_query(Some(&format!("perf={label}")));
                                    let _ = window.navigate(url);
                                }
                            }
                        }
                        // Development: a second window (`settings`, `setup`, `puzzle/<id>`)
                        // opened at start, to test one without the menu.
                        if let Ok(route) = std::env::var("SIDECOACH_OPEN_WINDOW") {
                            let _ =
                                tauri::async_runtime::block_on(open_window(handle.clone(), route));
                        }
                    }
                    Err(err) => {
                        eprintln!("sidecar: {err}");
                        *state.error.lock().unwrap() = Some(err.clone());
                        if let Some(window) = handle.get_webview_window("main") {
                            let _ = window.set_title(&format!("Sidecoach: {err}"));
                        }
                        if !splash {
                            show_main(&handle);
                        }
                    }
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            // Sidecoach's own panel in front: the captions step aside, as the phone's do.
            if let WindowEvent::Focused(focused) = event {
                if window.label() == "main" {
                    let _ = window
                        .app_handle()
                        .emit_to("w-captions", "panel-front", *focused);
                }
            }
            // The splash closed by the person (Alt+F4, the taskbar) before the main window
            // showed: nothing would be left on the screen and the app ran on unseen (setup
            // finding 13). It quits, as closing the main window does. `splash_done` shows the
            // main window before it closes the splash, so that close is not this one; with
            // the tray setting the tray icon is still there to reopen it.
            if let WindowEvent::CloseRequested { .. } = event {
                if window.label() == "splash" {
                    let app = window.app_handle();
                    let main_shown = app
                        .get_webview_window("main")
                        .and_then(|w| w.is_visible().ok())
                        .unwrap_or(false);
                    let tray = app.state::<Shell>().settings.lock().unwrap().close == "tray";
                    if !main_shown && !tray {
                        app.exit(0);
                    }
                }
            }
            // Where each window is, for the next start (`places`).
            match event {
                WindowEvent::Moved(_) | WindowEvent::Resized(_) => places::record(window),
                WindowEvent::CloseRequested { .. } => {
                    places::record(window);
                    places::save(window.app_handle());
                }
                _ => {}
            }
            // The main window closing quits, every other window with it -- the captions',
            // hidden, would keep the app and its sidecar running with nothing on the screen --
            // unless the setting says the tray; the other windows just close.
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    let close = window
                        .app_handle()
                        .state::<Shell>()
                        .settings
                        .lock()
                        .unwrap()
                        .close
                        .clone();
                    if close == "tray" {
                        api.prevent_close();
                        let _ = window.hide();
                    } else {
                        window.app_handle().exit(0);
                    }
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building the Sidecoach shell");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            places::save(handle);
            stop_sidecar(handle, &handle.state::<Sidecar>());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{child_exited, keep_previous_log};
    use std::process::{Command, Stdio};

    #[test]
    fn stopped_means_the_process_ended() {
        // Nothing held: not an exit.
        assert!(!child_exited(&mut None));
        // A process that is alive (it waits on its stdin) is not stopped, however long.
        #[cfg(windows)]
        let mut waiting = Command::new("cmd");
        #[cfg(windows)]
        waiting.args(["/c", "pause"]);
        #[cfg(not(windows))]
        let mut waiting = Command::new("cat");
        let mut alive = waiting
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .spawn()
            .unwrap();
        let mut held = Some(alive);
        assert!(!child_exited(&mut held));
        alive = held.take().unwrap();
        alive.kill().unwrap();
        alive.wait().unwrap();
        // Once ended, it says so, and goes on saying so.
        let mut held = Some(alive);
        assert!(child_exited(&mut held));
        assert!(child_exited(&mut held));
    }

    #[test]
    fn the_previous_launch_log_is_kept() {
        let dir = std::env::temp_dir().join(format!("sidecoach-log-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sidecar.log"), "the crash").unwrap();
        keep_previous_log(&dir);
        assert!(!dir.join("sidecar.log").exists());
        assert_eq!(
            std::fs::read_to_string(dir.join("sidecar.1.log")).unwrap(),
            "the crash"
        );
        // The one before that is replaced; no log at all is no error.
        std::fs::write(dir.join("sidecar.log"), "the next").unwrap();
        keep_previous_log(&dir);
        assert_eq!(
            std::fs::read_to_string(dir.join("sidecar.1.log")).unwrap(),
            "the next"
        );
        keep_previous_log(&dir);
        let _ = std::fs::remove_file(dir.join("sidecar.1.log"));
        let _ = std::fs::remove_dir(&dir);
    }
}
