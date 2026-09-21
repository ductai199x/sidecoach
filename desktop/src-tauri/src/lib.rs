//! The Sidecoach desktop shell: a window, a menu bar and a tray around the Python
//! pipeline, which runs as a sidecar behind its own HTTP API (docs/app-plan.md).
//!
//! The shell knows nothing about chess. It spawns the sidecar (`sidecoach panel --port 0
//! --announce --parent-pid <us> --no-persist --no-record`), reads the one JSON line the
//! sidecar prints with its port, waits for `/api/health`, hands the port and a bearer
//! token to the page, and kills the sidecar on exit; the sidecar also exits by itself when
//! this process is gone. The menu bar and the tray are built from the page with the JS
//! API (desktop/src/lib/shell.svelte.ts), so they stay one implementation with the
//! browser's header. Closing the main window hides it to the tray or quits, as the
//! shell's own setting says (Tai, 2026-09-21); the puzzle and settings pages open in
//! windows of their own.
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
use tauri::{AppHandle, Manager, RunEvent, State, WebviewUrl, WebviewWindowBuilder, WindowEvent};

/// How long the shell waits for the sidecar's port line and for its health route.
const START_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Default)]
struct Sidecar {
    child: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
    token: String,
}

/// The shell's own settings, kept apart from the sidecar's TOML because the shell needs
/// them before the sidecar is up: what closing the window does, and start with the OS.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct ShellSettings {
    close: String, // "tray" | "quit"
    autostart: bool,
}

impl Default for ShellSettings {
    fn default() -> Self {
        Self { close: "tray".into(), autostart: false }
    }
}

#[derive(Default)]
struct Shell {
    settings: Mutex<ShellSettings>,
}

fn settings_path(app: &AppHandle) -> Option<PathBuf> {
    app.path().app_config_dir().ok().map(|d| d.join("shell.json"))
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
    starts.push(std::env::current_exe().ok().and_then(|p| p.parent().map(PathBuf::from)));
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

fn embedded_python(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .resource_dir()
        .ok()
        .map(|r| r.join("python").join(if cfg!(windows) { "python.exe" } else { "bin/python3" }))
        .filter(|p| p.exists())
}

/// The command that runs the sidecar: the embedded interpreter when the bundle carries
/// one, else `uv run` in the repository (development).
fn sidecar_command(app: &AppHandle, token: &str) -> Result<Command, String> {
    let mut cmd = if let Some(python) = embedded_python(app) {
        let mut c = Command::new(python);
        c.args(["-m", "sidecoach", "panel"]);
        c
    } else {
        let root = repo_root().ok_or("no embedded Python and no repository root (pyproject.toml) found")?;
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
        cmd.args(["--token", token]);
    }
    if let Some(dir) = data_dir(app) {
        let _ = std::fs::create_dir_all(&dir);
        cmd.env("SIDECOACH_DATA_DIR", dir);
    }
    if let Ok(extra) = std::env::var("SIDECOACH_SIDECAR_ARGS") {
        cmd.args(extra.split_whitespace());
    }
    Ok(cmd)
}

/// One HTTP GET to the sidecar's health route; true on a 200.
fn healthy(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}");
    let Ok(mut stream) = TcpStream::connect_timeout(&addr.parse().expect("address"), Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream.write_all(b"GET /api/health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n").is_err() {
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
        .and_then(|d| std::fs::File::create(d.join("sidecar.log")).ok())
        .map(Stdio::from)
        .unwrap_or_else(Stdio::inherit);
    cmd.stdout(Stdio::piped()).stderr(stderr).stdin(Stdio::null());
    let mut child = cmd.spawn().map_err(|e| format!("cannot start the sidecar: {e}"))?;
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
            return Err(format!("the sidecar on port {port} never answered /api/health"));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    *state.port.lock().unwrap() = Some(port);
    Ok(port)
}

fn stop_sidecar(state: &Sidecar) {
    if let Some(mut child) = state.child.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn random_token() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut x = nanos as u64 ^ (std::process::id() as u64).rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = String::new();
    for _ in 0..4 {
        // xorshift: enough for a per-launch secret only local processes could read anyway
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        out.push_str(&format!("{x:016x}"));
    }
    out
}

// ---- what the page can ask -----------------------------------------------------------------

#[tauri::command]
fn sidecar_port(state: State<'_, Sidecar>) -> Option<u16> {
    *state.port.lock().unwrap()
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
fn set_shell_settings(app: AppHandle, shell: State<'_, Shell>, settings: ShellSettingsPatch) -> ShellSettings {
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

/// A page of the app in a window of its own: `settings`, `setup`, or `puzzle/<id>`. A
/// second request for the same route brings the window it has to the front.
#[tauri::command]
fn open_window(app: AppHandle, route: String) -> Result<(), String> {
    let route = route.trim_start_matches('/').to_string();
    let label: String = format!("w-{}", route.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect::<String>());
    if let Some(w) = app.get_webview_window(&label) {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
        return Ok(());
    }
    let (title, width, height) = if route.starts_with("puzzle/") {
        (format!("Sidecoach · {}", route.replace("puzzle/", "puzzle ")), 1180.0, 820.0)
    } else if route == "settings" {
        ("Sidecoach · Settings".to_string(), 900.0, 760.0)
    } else {
        ("Sidecoach".to_string(), 900.0, 700.0)
    };
    WebviewWindowBuilder::new(&app, &label, WebviewUrl::App(route.into()))
        .title(title)
        .inner_size(width, height)
        .min_inner_size(600.0, 420.0)
        .build()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let sidecar = Sidecar { token: random_token(), ..Default::default() };
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(sidecar)
        .manage(Shell::default())
        .invoke_handler(tauri::generate_handler![
            sidecar_port,
            sidecar_token,
            data_dir_command,
            shell_settings,
            set_shell_settings,
            open_window
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            *handle.state::<Shell>().settings.lock().unwrap() = load_shell_settings(&handle);
            std::thread::spawn(move || {
                let state = handle.state::<Sidecar>();
                match start_sidecar(&handle, &state) {
                    Ok(port) => {
                        // Development (the panel still served by the sidecar) may ask for
                        // the window to be pointed at it; the product keeps its own page.
                        if let Ok(query) = std::env::var("SIDECOACH_PAGE_QUERY") {
                            if let Some(window) = handle.get_webview_window("main") {
                                if let Ok(parsed) = format!("http://127.0.0.1:{port}/{query}").parse() {
                                    let _ = window.navigate(parsed);
                                }
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("sidecar: {err}");
                        if let Some(window) = handle.get_webview_window("main") {
                            let _ = window.set_title(&format!("Sidecoach: {err}"));
                        }
                    }
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            // The main window closing hides to the tray unless the setting says quit; the
            // other windows just close.
            if let WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    let close = window.app_handle().state::<Shell>().settings.lock().unwrap().close.clone();
                    if close == "tray" {
                        api.prevent_close();
                        let _ = window.hide();
                    }
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building the Sidecoach shell");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            stop_sidecar(&handle.state::<Sidecar>());
        }
    });
}
