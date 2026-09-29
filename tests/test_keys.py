"""Key input degrades to nothing without a terminal."""

import io
import sys

import pytest

from sidecoach.ui.keys import KeyReader


@pytest.mark.parametrize("stdin", [io.StringIO("q"), None], ids=["a-pipe", "no-stdin"])
def test_key_reader_without_a_terminal_reports_no_keys(
    stdin: io.StringIO | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "stdin", stdin)
    with KeyReader() as keys:
        assert not keys.enabled and keys.read() is None
