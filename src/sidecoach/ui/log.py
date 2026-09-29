"""The panel's debug log: structured records, a ring buffer for the page, a JSONL file for later.

A record is terse on the line and rich underneath: `text` is what the terminal shows, `data`
is what a click expands (the full assessment, the rejected drafts, the PV). Records are
tagged with a channel -- where they came from -- and a level, so the page can show the
important channels at `info` and keep the per-frame detail behind a verbose toggle.

Every record is also appended to a JSONL file when one is given, so a session can be read
back after the fact; the terminal is only a view over that file.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

LEVELS = ("debug", "info", "warn", "error")


@dataclass
class LogRecord:
    seq: int
    t: float  # seconds since the session began
    chan: str
    level: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DebugLog:
    """Thread-safe ring of records, mirrored to a JSONL file when `path` is given."""

    def __init__(self, path: Path | None = None, maxlen: int = 3000) -> None:
        self._records: deque[LogRecord] = deque(maxlen=maxlen)
        self._seq = 0
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._file: IO[str] | None = None
        self.path: Path | None = None
        if path is not None:
            self.attach(path)

    def attach(self, path: Path) -> None:
        """Start mirroring to `path`, writing out whatever was logged before the file existed
        (the session folder is only known once the engine starts)."""
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if self._file is not None:
                self._file.close()
            self._file = path.open("a", encoding="utf-8")
            self.path = path
            for record in self._records:
                self._write(record)

    def detach(self) -> None:
        """Stop mirroring to a file (a recording that ended mid-session)."""
        with self._lock:
            if self._file is not None:
                self._file.close()
            self._file = None
            self.path = None

    def _write(self, record: LogRecord) -> None:
        if self._file is None:
            return
        try:
            self._file.write(json.dumps(record.as_dict(), default=str) + "\n")
            self._file.flush()
        except OSError:
            self._file = None  # a full disk costs the file, never the session

    def emit(self, chan: str, level: str, text: str, **data: Any) -> LogRecord:
        with self._lock:
            self._seq += 1
            record = LogRecord(
                self._seq, round(time.monotonic() - self._t0, 3), chan, level, text, data
            )
            self._records.append(record)
            self._write(record)
        return record

    @property
    def cursor(self) -> int:
        return self._seq

    def since(self, seq: int) -> list[dict[str, Any]]:
        with self._lock:
            return [r.as_dict() for r in self._records if r.seq > seq]

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def elide(text: str, limit: int = 80) -> str:
    """The first `limit` characters of a line, with an ellipsis when it was cut."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
