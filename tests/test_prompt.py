"""The system message: its parts in order, and C appended only when it has content."""

from __future__ import annotations

from pathlib import Path

import chess

from sidecoach.coach import prompt
from sidecoach.coach.conversation import Conversation
from sidecoach.events import GameStarted, Side


def test_the_system_message_is_persona_contract_rules_and_reply_format_in_order() -> None:
    text = prompt.system_message(with_corrections=False)
    parts = (prompt.PERSONA, prompt.CONTRACT, prompt.RULES, prompt.REPLY_FORMAT)
    order = [text.index(part) for part in parts]
    assert order == sorted(order)
    # The contract tells the model what each message is, and where the course of the game is.
    assert "Every move on the board" in prompt.CONTRACT
    assert "takeback" in prompt.CONTRACT and "re-read" in prompt.CONTRACT
    assert "the verdicts and the tally, and nowhere else" in prompt.CONTRACT
    assert "brilliant, great, best, excellent, good, book" in prompt.CONTRACT
    assert "Rules learned from earlier games" not in text


def test_corrections_are_appended_only_when_the_file_has_content(
    tmp_path: Path, monkeypatch
) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(prompt, "CORRECTIONS_PATH", empty)
    prompt.corrections.cache_clear()
    assert "Rules learned from earlier games" not in prompt.system_message()

    filled = tmp_path / "c.md"
    filled.write_text("## 1\nThe coach said X; the record showed Y; say Y.\n", encoding="utf-8")
    monkeypatch.setattr(prompt, "CORRECTIONS_PATH", filled)
    prompt.corrections.cache_clear()
    text = prompt.system_message()
    assert text.endswith("as the rules above.\n## 1\nThe coach said X; the record showed Y; say Y.")
    assert text.index(prompt.REPLY_FORMAT) < text.index("Rules learned from earlier games")
    prompt.corrections.cache_clear()

    missing = tmp_path / "missing.md"
    monkeypatch.setattr(prompt, "CORRECTIONS_PATH", missing)
    prompt.corrections.cache_clear()
    assert "Rules learned from earlier games" not in prompt.system_message()
    prompt.corrections.cache_clear()


def test_the_shipped_corrections_file_exists_and_is_read() -> None:
    assert prompt.CORRECTIONS_PATH.exists()
    prompt.corrections.cache_clear()
    assert prompt.corrections() == prompt.CORRECTIONS_PATH.read_text(encoding="utf-8").strip()


def test_the_new_game_message_states_the_side_and_never_a_rating() -> None:
    """Whoever is playing gets the same coach: no rating is stated or pitched to."""
    convo = Conversation()
    convo.new_game(GameStarted(game_id="g", user_side=Side.BLACK, start_fen=chess.Board().fen()))
    first = convo.messages()[0]["content"]
    assert first.startswith("New game. You coach the player, who is Black, against a bot.")
    assert "rated" not in first
    assert "rating" not in prompt.system_message(with_corrections=False).lower()
