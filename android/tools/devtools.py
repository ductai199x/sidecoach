"""Run a line of JavaScript in the app's WebView and print what it answers.

The panel on the phone is the desktop's own page in a WebView, and a WebView with
`setWebContentsDebuggingEnabled` speaks the same DevTools protocol Chrome does. This
forwards its socket and sends one `Runtime.evaluate`, so the page can be inspected from here
the way `chrome://inspect` inspects it -- what the DOM came to, what a store holds, what a
fetch answered.

    adb forward tcp:9222 localabstract:webview_devtools_remote_<pid>   (done for you)
    uv run --no-sync python android/tools/devtools.py "document.title"
    uv run --no-sync python android/tools/devtools.py --file probe.js

A WebSocket client in sixty lines rather than a dependency: the protocol's handshake is an
HTTP upgrade and its frames are short, and nothing here needs more than one round trip.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import struct
import subprocess
import sys
import urllib.request

DEVTOOLS_PORT = 9222
PACKAGE = "com.sidecoach"


def adb() -> str:
    return os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe")


def forward(serial: str) -> None:
    pid = subprocess.run(
        [adb(), "-s", serial, "shell", "pidof", PACKAGE],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not pid:
        raise SystemExit(f"{PACKAGE} is not running on {serial}")
    subprocess.run(
        [
            adb(),
            "-s",
            serial,
            "forward",
            f"tcp:{DEVTOOLS_PORT}",
            f"localabstract:webview_devtools_remote_{pid}",
        ],
        capture_output=True,
        check=True,
    )


def page_socket() -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{DEVTOOLS_PORT}/json", timeout=5) as r:
        pages = json.load(r)
    for page in pages:
        if page.get("type") == "page" and page.get("webSocketDebuggerUrl"):
            return page["webSocketDebuggerUrl"]
    raise SystemExit("no page in the WebView")


class Socket:
    """Just enough WebSocket for one request and one reply."""

    def __init__(self, url: str) -> None:
        match = re.match(r"ws://([^:/]+):(\d+)(/.*)", url)
        if not match:
            raise SystemExit(f"cannot read {url}")
        host, port, path = match.group(1), int(match.group(2)), match.group(3)
        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(request.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            head += self.sock.recv(1)
        if b"101" not in head.split(b"\r\n")[0]:
            raise SystemExit(f"the upgrade was refused: {head.splitlines()[0]!r}")
        self.rest = b""

    def send(self, text: str) -> None:
        payload = text.encode()
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        header = b"\x81"
        n = len(payload)
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 1 << 16:
            header += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.sock.sendall(header + mask + masked)

    def _read(self, n: int) -> bytes:
        while len(self.rest) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise SystemExit("the socket closed")
            self.rest += chunk
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def recv(self) -> str:
        first, second = self._read(2)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack(">H", self._read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self._read(8))[0]
        return self._read(length).decode("utf-8", "replace")

    def close(self) -> None:
        self.sock.close()


def evaluate(expression: str, serial: str = "emulator-5556") -> object:
    forward(serial)
    ws = Socket(page_socket())
    try:
        ws.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": expression,
                        "returnByValue": True,
                        "awaitPromise": True,
                    },
                }
            )
        )
        while True:
            message = json.loads(ws.recv())
            if message.get("id") == 1:
                return message.get("result")
    finally:
        ws.close()


def main() -> None:
    args = sys.argv[1:]
    serial = "emulator-5556"
    if args and args[0] == "--device":
        serial = args[1]
        args = args[2:]
    if args and args[0] == "--file":
        expression = open(args[1], encoding="utf-8").read()
    else:
        expression = " ".join(args) or "document.title"
    result = evaluate(expression, serial)
    print(json.dumps(result, indent=2)[:20000])


if __name__ == "__main__":
    main()
