"""Key input degrades to nothing without a terminal."""

import io
import sys

from sidecoach.ui.keys import KeyReader


def test_key_reader_without_a_terminal_reports_no_keys(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("q"))
    with KeyReader() as keys:
        assert not keys.enabled
        assert keys.read() is None


def test_key_reader_survives_a_missing_stdin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", None)
    with KeyReader() as keys:
        assert not keys.enabled and keys.read() is None
