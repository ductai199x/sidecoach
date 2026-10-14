"""Capture on a Wayland desktop: the screen through the desktop's screen-sharing portal.

Wayland lets no application read the screen by itself: an X11 grab through XWayland gets a
protocol error ("31 grabs failed in a row: X11 Protocol Error" on Ubuntu 24.04, 2026-10-13).
What every screen recorder does instead is ask the xdg-desktop-portal's ScreenCast interface:
the desktop asks the user once which screen to share, then hands over a PipeWire stream of it.
The answer is remembered (the portal's restore token, kept in the data folder), so a later
Start asks nothing.

The portal is spoken to over D-Bus with jeepney (pure Python; keyring brings it on Linux). The
stream is read with the system's GStreamer through ctypes -- `pipewiresrc` (PipeWire's own
plugin, on every desktop that ships PipeWire), `videoconvert` and `appsink` -- so nothing of
the app's own can disagree with the desktop's PipeWire.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import os
import secrets
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sidecoach.events import Region
from sidecoach.linux import wayland_session

__all__ = ["PortalCapture", "PortalCancelled", "PortalError", "portal_status", "wayland_session"]

TOKEN_FILE = "screencast-restore-token"

# org.freedesktop.portal.ScreenCast
SOURCE_MONITOR = 1
CURSOR_HIDDEN = 1
PERSIST_UNTIL_REVOKED = 2
# org.freedesktop.portal.Request.Response
RESPONSE_OK = 0
RESPONSE_CANCELLED = 1

# The user picks a screen in the desktop's own dialog: no deadline a person could miss.
ANSWER_TIMEOUT_S = 600.0
FIRST_FRAME_TIMEOUT_S = 10.0
# Where a desktop keeps its GStreamer, preferred over whatever an AppImage put on the path.
SYSTEM_LIB_DIRS = (
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/lib/x86_64-linux-gnu",
)


class PortalError(RuntimeError):
    """The portal refused, or the stream cannot be read; the message says what to do."""


class PortalCancelled(PortalError):
    """The user closed the desktop's screen-sharing dialog without sharing. Final: the
    capture loop does not retry it (`fatal`), or every retry would ask again."""

    fatal = True


def request_path(unique_name: str, token: str) -> str:
    """The object path of the Request a portal call with `handle_token` answers on."""
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"/org/freedesktop/portal/desktop/request/{sender}/{token}"


def stream_of(results: Mapping[str, Any]) -> tuple[int, tuple[int, int] | None]:
    """The first stream of a Start response: its PipeWire node and, when the portal says, its
    position on the desktop."""
    streams = _value(results.get("streams")) or []
    if not streams:
        raise PortalError("the desktop shared no screen")
    node, props = streams[0]
    position = _value(props.get("position")) if isinstance(props, Mapping) else None
    return int(node), (int(position[0]), int(position[1])) if position else None


def _value(variant: Any) -> Any:
    """A D-Bus variant's value as jeepney gives it, (signature, value)."""
    if isinstance(variant, tuple) and len(variant) == 2 and isinstance(variant[0], str):
        return variant[1]
    return variant


def read_token(state_dir: Path | None) -> str | None:
    if state_dir is None:
        return None
    try:
        token = (state_dir / TOKEN_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def write_token(state_dir: Path | None, token: str | None) -> None:
    if state_dir is None:
        return
    path = state_dir / TOKEN_FILE
    with contextlib.suppress(OSError):
        if token:
            state_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(token, encoding="utf-8")
        else:
            path.unlink(missing_ok=True)


class _Portal:
    """One ScreenCast session on the session bus. The connection stays open while the stream
    is read: closing it ends the session."""

    def __init__(self) -> None:
        from jeepney import DBusAddress  # type: ignore[reportMissingImports]
        from jeepney.io.blocking import open_dbus_connection  # type: ignore[reportMissingImports]

        try:
            self.conn = open_dbus_connection(bus="SESSION", enable_fds=True)
        except Exception as exc:  # noqa: BLE001
            raise PortalError(f"no session bus to ask the desktop for the screen: {exc}") from exc
        self.address = DBusAddress(
            "/org/freedesktop/portal/desktop",
            bus_name="org.freedesktop.portal.Desktop",
            interface="org.freedesktop.portal.ScreenCast",
        )
        self.session: str | None = None
        self._tokens = itertools.count(1)

    def property(self, name: str) -> Any:
        from jeepney import (  # type: ignore[reportMissingImports]
            DBusAddress,
            MessageType,
            new_method_call,
        )

        props = DBusAddress(
            self.address.object_path,
            bus_name=self.address.bus_name,
            interface="org.freedesktop.DBus.Properties",
        )
        reply = self.conn.send_and_get_reply(
            new_method_call(props, "Get", "ss", (self.address.interface, name)), timeout=10
        )
        if reply.header.message_type == MessageType.error:
            raise PortalError(
                "this desktop has no screen-sharing portal (xdg-desktop-portal): "
                f"{_error_text(reply)}"
            )
        return _value(reply.body[0])

    def call(
        self, method: str, signature: str, args: tuple[Any, ...], options: dict[str, Any]
    ) -> dict[str, Any]:
        """A portal call that answers through a Request's Response signal: its results."""
        from jeepney import (  # type: ignore[reportMissingImports]
            MatchRule,
            MessageType,
            new_method_call,
        )
        from jeepney.bus_messages import message_bus  # type: ignore[reportMissingImports]

        token = f"sidecoach{next(self._tokens)}_{secrets.token_hex(4)}"
        options = {**options, "handle_token": ("s", token)}
        rule = MatchRule(
            type="signal",
            interface="org.freedesktop.portal.Request",
            member="Response",
            path=request_path(self.conn.unique_name, token),
        )
        self.conn.send_and_get_reply(message_bus.AddMatch(rule), timeout=10)
        with self.conn.filter(rule) as queue:
            reply = self.conn.send_and_get_reply(
                new_method_call(self.address, method, signature, (*args, options)), timeout=30
            )
            if reply.header.message_type == MessageType.error:
                raise PortalError(f"the desktop refused {method}: {_error_text(reply)}")
            try:
                answer = self.conn.recv_until_filtered(queue, timeout=ANSWER_TIMEOUT_S)
            except TimeoutError as exc:
                raise PortalError("the desktop's screen-sharing dialog was not answered") from exc
        response, results = answer.body
        if response == RESPONSE_CANCELLED:
            raise PortalCancelled("screen sharing was cancelled in the desktop's dialog")
        if response != RESPONSE_OK:
            raise PortalError(f"the desktop did not share the screen ({method}: {response})")
        return dict(results)

    def start(self, restore_token: str | None) -> tuple[int, tuple[int, int] | None, str | None]:
        """Ask for one monitor: its PipeWire node, its position, and the token that shares it
        again without asking."""
        version = int(self.property("version") or 0)
        cursors = int(self.property("AvailableCursorModes") or 0)
        created = self.call(
            "CreateSession",
            "a{sv}",
            (),
            {"session_handle_token": ("s", f"sidecoach_{secrets.token_hex(4)}")},
        )
        self.session = str(_value(created.get("session_handle")))
        options: dict[str, Any] = {
            "types": ("u", SOURCE_MONITOR),
            "multiple": ("b", False),
        }
        if cursors & CURSOR_HIDDEN:
            # The pointer over a square is no piece: kept out of the frames.
            options["cursor_mode"] = ("u", CURSOR_HIDDEN)
        if version >= 4:
            options["persist_mode"] = ("u", PERSIST_UNTIL_REVOKED)
            if restore_token:
                options["restore_token"] = ("s", restore_token)
        self.call("SelectSources", "oa{sv}", (self.session,), options)
        started = self.call("Start", "osa{sv}", (self.session, ""), {})
        node, position = stream_of(started)
        token = _value(started.get("restore_token"))
        return node, position, str(token) if token else None

    def pipewire_fd(self) -> int:
        from jeepney import MessageType, new_method_call  # type: ignore[reportMissingImports]

        reply = self.conn.send_and_get_reply(
            new_method_call(self.address, "OpenPipeWireRemote", "oa{sv}", (self.session, {})),
            timeout=10,
        )
        if reply.header.message_type == MessageType.error:
            raise PortalError(f"the desktop gave no PipeWire stream: {_error_text(reply)}")
        return int(reply.body[0].to_raw_fd())

    def close(self) -> None:
        from jeepney import DBusAddress, new_method_call  # type: ignore[reportMissingImports]

        if self.session:
            closing = DBusAddress(
                self.session,
                bus_name=self.address.bus_name,
                interface="org.freedesktop.portal.Session",
            )
            with contextlib.suppress(Exception):
                self.conn.send_message(new_method_call(closing, "Close"))
            self.session = None
        with contextlib.suppress(Exception):
            self.conn.close()


def _error_text(reply: Any) -> str:
    from jeepney import HeaderFields  # type: ignore[reportMissingImports]

    name = reply.header.fields.get(HeaderFields.error_name, "")
    body = reply.body[0] if reply.body else ""
    return f"{name} {body}".strip()


class _GstMapInfo(ctypes.Structure):
    _fields_ = [
        ("memory", ctypes.c_void_p),
        ("flags", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
        ("maxsize", ctypes.c_size_t),
        ("user_data", ctypes.c_void_p * 4),
        ("_gst_reserved", ctypes.c_void_p * 4),
    ]


class _GError(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("code", ctypes.c_int), ("message", ctypes.c_char_p)]


GST_STATE_NULL = 1
GST_STATE_PLAYING = 4
GST_MAP_READ = 1
GST_MESSAGE_EOS = 1 << 0
GST_MESSAGE_ERROR = 1 << 1

_gst_lock = threading.Lock()
_gst: tuple[Any, Any] | None = None


def _system_library(name: str) -> Any:
    for folder in SYSTEM_LIB_DIRS:
        path = Path(folder) / name
        if path.exists():
            return ctypes.CDLL(str(path))
    return ctypes.CDLL(name)


def _without_bundled_plugins() -> None:
    """An AppImage points GStreamer at its own plugins (GST_PLUGIN_SYSTEM_PATH...), where
    there is no `pipewiresrc`; the desktop's are the ones that match its PipeWire."""
    appdir = os.environ.get("APPDIR")
    if not appdir:
        return
    for name in list(os.environ):
        if name.startswith("GST_") and appdir in os.environ.get(name, ""):
            os.environ.pop(name, None)


def gstreamer() -> tuple[Any, Any]:
    """The system's libgstreamer and libgstapp, initialised once per process."""
    global _gst
    with _gst_lock:
        if _gst is not None:
            return _gst
        _without_bundled_plugins()
        try:
            gst = _system_library("libgstreamer-1.0.so.0")
            app = _system_library("libgstapp-1.0.so.0")
        except OSError as exc:
            raise PortalError(
                "GStreamer is not installed: it reads the shared screen "
                "(install gstreamer1.0-pipewire and gstreamer1.0-plugins-base)"
            ) from exc
        vp, cp, ci, ui, u64 = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint64,
        )
        signatures: dict[str, tuple[Any, list[Any]]] = {
            "gst_init": (None, [vp, vp]),
            "gst_parse_launch": (vp, [cp, ctypes.POINTER(ctypes.POINTER(_GError))]),
            "gst_bin_get_by_name": (vp, [vp, cp]),
            "gst_element_set_state": (ci, [vp, ci]),
            "gst_element_get_bus": (vp, [vp]),
            "gst_bus_pop_filtered": (vp, [vp, ci]),
            "gst_sample_get_buffer": (vp, [vp]),
            "gst_sample_get_caps": (vp, [vp]),
            "gst_caps_get_structure": (vp, [vp, ui]),
            "gst_structure_get_int": (ci, [vp, cp, ctypes.POINTER(ci)]),
            "gst_buffer_map": (ci, [vp, ctypes.POINTER(_GstMapInfo), ci]),
            "gst_buffer_unmap": (None, [vp, ctypes.POINTER(_GstMapInfo)]),
            "gst_mini_object_unref": (None, [vp]),
            "gst_object_unref": (None, [vp]),
            "gst_element_factory_find": (vp, [cp]),
        }
        for fn, (restype, argtypes) in signatures.items():
            f = getattr(gst, fn)
            f.restype, f.argtypes = restype, argtypes
        app.gst_app_sink_try_pull_sample.restype = vp
        app.gst_app_sink_try_pull_sample.argtypes = [vp, u64]
        gst.gst_init(None, None)
        _gst = (gst, app)
        return _gst


def missing_elements() -> list[str]:
    """The GStreamer elements the stream needs that this desktop lacks, by package."""
    gst, _ = gstreamer()
    missing = []
    for element, package in (
        ("pipewiresrc", "gstreamer1.0-pipewire"),
        ("videoconvert", "gstreamer1.0-plugins-base"),
        ("appsink", "gstreamer1.0-plugins-base"),
    ):
        factory = gst.gst_element_factory_find(element.encode())
        if factory:
            gst.gst_object_unref(factory)
        else:
            missing.append(f"{element} ({package})")
    return missing


class _Stream:
    """A PipeWire stream read by a GStreamer pipeline; the latest frame kept as BGR."""

    def __init__(self, fd: int, node: int) -> None:
        self._gst, self._app = gstreamer()
        lacking = missing_elements()
        if lacking:
            raise PortalError("this desktop lacks GStreamer's " + ", ".join(lacking))
        description = (
            f"pipewiresrc fd={fd} path={node} always-copy=true do-timestamp=true ! "
            "videoconvert ! video/x-raw,format=BGRx ! "
            "appsink name=sink max-buffers=1 drop=true sync=false"
        )
        error = ctypes.POINTER(_GError)()
        self._pipeline = self._gst.gst_parse_launch(description.encode(), ctypes.byref(error))
        if not self._pipeline:
            message = error.contents.message.decode() if error else "unknown"
            raise PortalError(f"the shared screen cannot be read: {message}")
        self._sink = self._gst.gst_bin_get_by_name(self._pipeline, b"sink")
        self._bus = self._gst.gst_element_get_bus(self._pipeline)
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._first = threading.Event()
        self._stop = threading.Event()
        self.failure: str | None = None
        self._gst.gst_element_set_state(self._pipeline, GST_STATE_PLAYING)
        self._thread = threading.Thread(target=self._read, name="portal-stream", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        gst, app = self._gst, self._app
        while not self._stop.is_set():
            sample = app.gst_app_sink_try_pull_sample(self._sink, 100_000_000)  # 100 ms
            if not sample:
                message = gst.gst_bus_pop_filtered(self._bus, GST_MESSAGE_ERROR | GST_MESSAGE_EOS)
                if message:
                    gst.gst_mini_object_unref(message)
                    # Sharing stopped from the desktop's indicator, or the screen went away.
                    self.failure = "the desktop stopped sharing the screen"
                    self._first.set()
                    return
                continue
            try:
                frame = self._to_bgr(sample)
            finally:
                gst.gst_mini_object_unref(sample)
            if frame is not None:
                with self._lock:
                    self._frame = frame
                self._first.set()

    def _to_bgr(self, sample: int) -> np.ndarray | None:
        gst = self._gst
        caps = gst.gst_sample_get_caps(sample)
        structure = gst.gst_caps_get_structure(caps, 0) if caps else None
        width, height = ctypes.c_int(0), ctypes.c_int(0)
        if not structure:
            return None
        gst.gst_structure_get_int(structure, b"width", ctypes.byref(width))
        gst.gst_structure_get_int(structure, b"height", ctypes.byref(height))
        w, h = width.value, height.value
        buffer = gst.gst_sample_get_buffer(sample)
        if not buffer or w <= 0 or h <= 0:
            return None
        info = _GstMapInfo()
        if not gst.gst_buffer_map(buffer, ctypes.byref(info), GST_MAP_READ):
            return None
        try:
            stride = info.size // h
            if stride < w * 4:
                return None
            raw = np.ctypeslib.as_array(info.data, shape=(stride * h,))
            bgrx = raw.reshape(h, stride)[:, : w * 4].reshape(h, w, 4)
            return np.ascontiguousarray(bgrx[:, :, :3])
        finally:
            gst.gst_buffer_unmap(buffer, ctypes.byref(info))

    def latest(self, timeout: float) -> np.ndarray:
        if not self._first.wait(timeout):
            raise PortalError("the shared screen sends no picture")
        if self.failure:
            raise PortalError(self.failure)
        with self._lock:
            frame = self._frame
        if frame is None:
            raise PortalError("the shared screen sends no picture")
        return frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        gst = self._gst
        gst.gst_element_set_state(self._pipeline, GST_STATE_NULL)
        for obj in (self._sink, self._bus, self._pipeline):
            if obj:
                gst.gst_object_unref(obj)
        self._sink = self._bus = self._pipeline = None


class PortalCapture:
    """The screen of a Wayland desktop, shared through the portal (see the module)."""

    name = "portal"

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir
        self._portal: _Portal | None = None
        self._stream: _Stream | None = None
        self._origin = (0, 0)
        self._refused: PortalCancelled | None = None
        self._open()

    def _open(self) -> None:
        portal = _Portal()
        try:
            node, position, token = portal.start(read_token(self._state_dir))
            # A token shares once: the next Start needs the one this answer brought.
            write_token(self._state_dir, token)
            stream = _Stream(portal.pipewire_fd(), node)
            try:
                stream.latest(FIRST_FRAME_TIMEOUT_S)
            except Exception:
                stream.close()
                raise
        except Exception:
            portal.close()
            raise
        self._portal, self._stream = portal, stream
        self._origin = position or (0, 0)

    def monitors(self) -> list[Region]:
        assert self._stream is not None
        frame = self._stream.latest(FIRST_FRAME_TIMEOUT_S)
        return [Region(self._origin[0], self._origin[1], frame.shape[1], frame.shape[0])]

    def grab(self, region: Region) -> np.ndarray:
        if self._refused is not None:
            raise self._refused
        if self._stream is None:
            raise PortalError("the screen is not shared")
        frame = self._stream.latest(FIRST_FRAME_TIMEOUT_S)
        x0, y0 = region.x - self._origin[0], region.y - self._origin[1]
        if x0 < 0 or y0 < 0 or x0 + region.w > frame.shape[1] or y0 + region.h > frame.shape[0]:
            raise PortalError(
                f"the shared screen is {frame.shape[1]}x{frame.shape[0]}, "
                f"not {region.w}x{region.h} any more"
            )
        return frame[y0 : y0 + region.h, x0 : x0 + region.w]

    def reopen(self) -> None:
        if self._refused is not None:
            return  # the user said no: asked once, not again
        self.close()
        # Again through the portal, with the token this session's Start saved: no dialog.
        try:
            self._open()
        except PortalCancelled as exc:
            self._refused = exc
            raise

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._portal is not None:
            self._portal.close()
            self._portal = None


def portal_status() -> tuple[str, str]:
    """The doctor's word on Wayland capture, without asking the user anything: the portal's
    version and the GStreamer elements the stream needs."""
    try:
        portal = _Portal()
        try:
            version = int(portal.property("version") or 0)
            sources = int(portal.property("AvailableSourceTypes") or 0)
        finally:
            portal.close()
        if not sources & SOURCE_MONITOR:
            return "error", "the desktop's screen-sharing portal cannot share a monitor"
        lacking = missing_elements()
    except PortalError as exc:
        return "error", str(exc)
    if lacking:
        return "error", "Wayland screen sharing needs " + ", ".join(lacking)
    remembered = "" if version >= 4 else " (asks at every Start: portal too old to remember)"
    return "ok", f"portal (Wayland screen sharing, version {version}){remembered}"
