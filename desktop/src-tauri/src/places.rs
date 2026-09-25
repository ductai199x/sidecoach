//! Where each window was: its size, its place on the screen and whether it was maximised,
//! kept beside the shell's settings (`window.json` in the app's config folder), so the app
//! opens as it was closed (Tai, 2026-09-25: "can the app on desktop remembers the window
//! size and dividers' positions so that when I open the app again it's the same as last time
//! I closed it?"). The page keeps its own dividers (its localStorage, `prefs`); the board it
//! fits into the window comes back at its size once the window does.
//!
//! One place per kind of window: the main one, the settings, and every puzzle window alike.
//! A place is recorded as the window moves or resizes (never while minimised, and while
//! maximised only the flag: the bounds to come back to are the ones before), and written when
//! a window closes and when the app exits. A place on no screen that is there now (a monitor
//! unplugged since) keeps its size and is centred.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, PhysicalPosition, PhysicalSize, WebviewWindow, Window};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Place {
    /// The outer top-left corner and the inner size, in physical pixels.
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub maximized: bool,
}

#[derive(Default)]
pub struct Places {
    map: Mutex<HashMap<String, Place>>,
    /// Windows restored while hidden whose maximise waits for them to be shown.
    maximize_on_show: Mutex<HashSet<String>>,
}

/// The kind of window a label names: its place is shared by every window of the kind.
fn kind(label: &str) -> Option<String> {
    match label {
        "main" => Some("main".into()),
        "splash" => None,
        _ if label.starts_with("w-puzzle-") => Some("puzzle".into()),
        _ => label.strip_prefix("w-").map(str::to_string),
    }
}

fn path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|d| d.join("window.json"))
}

/// The places written last time, read once at start.
pub fn load(app: &AppHandle) {
    let read: HashMap<String, Place> = path(app)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    *app.state::<Places>().map.lock().unwrap() = read;
}

pub fn save(app: &AppHandle) {
    let Some(p) = path(app) else { return };
    let places = app.state::<Places>();
    let text = {
        let map = places.map.lock().unwrap();
        serde_json::to_string_pretty(&*map).unwrap_or_default()
    };
    if let Some(dir) = p.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(p, text);
}

/// Where the window is now, as its kind's place.
pub fn record(window: &Window) {
    let Some(kind) = kind(window.label()) else {
        return;
    };
    if window.is_minimized().unwrap_or(false) || !window.is_visible().unwrap_or(false) {
        return;
    }
    let maximized = window.is_maximized().unwrap_or(false);
    let places = window.app_handle().state::<Places>();
    let mut map = places.map.lock().unwrap();
    let (Ok(pos), Ok(size)) = (window.outer_position(), window.inner_size()) else {
        return;
    };
    let now = Place {
        x: pos.x,
        y: pos.y,
        width: size.width,
        height: size.height,
        maximized,
    };
    let place = match (maximized, map.get(&kind)) {
        // Maximised: the flag, and the bounds from before, when there are any.
        (true, Some(before)) => Place {
            maximized: true,
            ..*before
        },
        _ => now,
    };
    map.insert(kind, place);
}

/// The window put back where its kind was, before it is shown. A maximised one is
/// maximised now if it is visible, else when `shown` is called for it.
pub fn restore(window: &WebviewWindow) {
    let Some(kind) = kind(window.label()) else {
        return;
    };
    let place = window
        .app_handle()
        .state::<Places>()
        .map
        .lock()
        .unwrap()
        .get(&kind)
        .copied();
    let Some(p) = place else { return };
    // The middle of the title bar must be on a screen that is there now.
    let (cx, cy) = (p.x + (p.width / 2) as i32, p.y + 16);
    let screen = window.available_monitors().ok().and_then(|monitors| {
        monitors.into_iter().find(|m| {
            let (mp, ms) = (m.position(), m.size());
            cx >= mp.x && cx < mp.x + ms.width as i32 && cy >= mp.y && cy < mp.y + ms.height as i32
        })
    });
    let _ = window.set_size(PhysicalSize::new(p.width, p.height));
    if screen.is_some() {
        let _ = window.set_position(PhysicalPosition::new(p.x, p.y));
    } else {
        let _ = window.center();
    }
    if p.maximized {
        if window.is_visible().unwrap_or(false) {
            let _ = window.maximize();
        } else {
            window
                .app_handle()
                .state::<Places>()
                .maximize_on_show
                .lock()
                .unwrap()
                .insert(window.label().to_string());
        }
    }
}

/// A window restored while hidden has just been shown: maximised now if it was.
pub fn shown(window: &WebviewWindow) {
    let waiting = window
        .app_handle()
        .state::<Places>()
        .maximize_on_show
        .lock()
        .unwrap()
        .remove(window.label());
    if waiting {
        let _ = window.maximize();
    }
}
