"""The system message: its parts in order, and C appended only when it has content."""

from __future__ import annotations

import re
from pathlib import Path

from sidecoach.coach import prompt
from sidecoach.events import Label


def test_the_system_message_is_persona_contract_rules_and_reply_format_in_order() -> None:
    text = prompt.system_message(with_corrections=False)
    parts = (prompt.PERSONA, prompt.CONTRACT, prompt.RULES, prompt.REPLY_FORMAT)
    order = [text.index(part) for part in parts]
    assert order == sorted(order)
    # The contract explains what the harness holds: its sections, the marks of who said what
    # and of an event, and every label a move's verdict can carry.
    for mark in (
        "# Setting",
        "# Game so far",
        "# Position now",
        "# Task",
        "PLAYER said:",
        "YOU (the coach) said:",
        "EVENT:",
    ):
        assert mark in prompt.CONTRACT, mark
    for label in Label:
        assert re.search(rf"\b{label.value}\b", prompt.CONTRACT), label
    assert "Lessons from earlier games" not in text, "no C without it"


def test_corrections_are_appended_only_when_the_file_has_content(
    tmp_path: Path, monkeypatch
) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(prompt, "CORRECTIONS_PATH", empty)
    prompt.corrections.cache_clear()
    assert "Lessons from earlier games" not in prompt.system_message()

    filled = tmp_path / "c.md"
    filled.write_text(
        "When asked about the opening: state its name as the message gives it.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt, "CORRECTIONS_PATH", filled)
    prompt.corrections.cache_clear()
    text = prompt.system_message()
    assert text.endswith("\nWhen asked about the opening: state its name as the message gives it.")
    assert text.index(prompt.REPLY_FORMAT) < text.index("Lessons from earlier games")
    prompt.corrections.cache_clear()

    missing = tmp_path / "missing.md"
    monkeypatch.setattr(prompt, "CORRECTIONS_PATH", missing)
    prompt.corrections.cache_clear()
    assert "Lessons from earlier games" not in prompt.system_message()
    prompt.corrections.cache_clear()


def test_the_shipped_corrections_file_exists_and_is_read() -> None:
    assert prompt.CORRECTIONS_PATH.exists()
    prompt.corrections.cache_clear()
    assert prompt.corrections() == prompt.CORRECTIONS_PATH.read_text(encoding="utf-8").strip()
