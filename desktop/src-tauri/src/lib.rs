//! The Sidecoach desktop shell: a window, a menu bar and a tray around the Python
//! pipeline, which runs as a sidecar behind its own HTTP API (docs/app-plan.md).
//!
//! The shell knows nothing about chess. It spawns the sidecar (`sidecoach panel --port 0
//! --announce --parent-pid <us> --no-persist --no-record`), reads the one JSON line the
//! sidecar prints with its port, waits for `/api/health`, points the window at it, and
//! kills it on exit; the sidecar also exits by itself when this process is gone. The menu
//! bar and the tray are built from the page with the JS API (web/shell.js), so they stay
//! one implementation with the browser's header.
//!
//! Development (`tauri dev`, or a debug build): the sidecar is `uv run --no-sync sidecoach
//! panel ...` in the repository root, found by walking up to `pyproject.toml`; extra
//! arguments come from `SIDECOACH_SIDECAR_ARGS` (a replay, a config). A release build runs
//! the embedded interpreter from the bundle's resources (Phase 1).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use tauri::{AppHandle, Manager, RunEvent, State};

/// How long the shell waits for the sidecar's port line and for its health route.
const START_TIMEOUT: Duration = Duration::from_secs(90);

#[derive(Default)]
struct Sidecar {
    child: Mutex<Option<Child>>,
    port: Mutex<Option<u16>>,
    token: String,
}

/// The repository root in development: the first ancestor of the working directory or
/// the executable with a `pyproject.toml`.
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

/// The command that runs the sidecar: the embedded interpreter when the bundle carries
/// one, else `uv run` in the repository (development).
fn sidecar_command(app: &AppHandle) -> Result<Command, String> {
    let embedded = app
        .path()
        .resource_dir()
        .ok()
        .map(|r| r.join("python").join(if cfg!(windows) { "python.exe" } else { "bin/python3" }))
        .filter(|p| p.exists());
    let mut cmd = if let Some(python) = embedded {
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
    if let Ok(extra) = std::env::var("SIDECOACH_SIDECAR_ARGS") {
        cmd.args(extra.split_whitespace());
    }
    Ok(cmd)
}

/// One HTTP GET to the sidecar's health route; true on a 200.
fn healthy(port: u16) -> bool {
    let addr = format!("127.0.0.1:{port}");
    let Ok(mut stream) = TcpStream::connect_timeout(
        &addr.parse().expect("address"),
        Duration::from_millis(500),
    ) else {
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
    let mut cmd = sidecar_command(app)?;
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit()).stdin(Stdio::null());
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

#[tauri::command]
fn sidecar_port(state: State<'_, Sidecar>) -> Option<u16> {
    *state.port.lock().unwrap()
}

#[tauri::command]
fn sidecar_token(state: State<'_, Sidecar>) -> String {
    state.token.clone()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(Sidecar::default())
        .invoke_handler(tauri::generate_handler![sidecar_port, sidecar_token])
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                let state = handle.state::<Sidecar>();
                match start_sidecar(&handle, &state) {
                    Ok(port) => {
                        let query = std::env::var("SIDECOACH_PAGE_QUERY").unwrap_or_default();
                        let url = format!("http://127.0.0.1:{port}/{query}");
                        if let Some(window) = handle.get_webview_window("main") {
                            if let Ok(parsed) = url.parse() {
                                let _ = window.navigate(parsed);
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
        .build(tauri::generate_context!())
        .expect("error while building the Sidecoach shell");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            stop_sidecar(&handle.state::<Sidecar>());
        }
    });
}
