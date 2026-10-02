"""The system message: its parts in order, and C appended only when it has content."""

from __future__ import annotations

import re
from pathlib import Path

from sidecoach.coach import prompt
from sidecoach.events import Label


def test_the_system_message_is_persona_contract_and_rules_in_order() -> None:
    text = prompt.system_message(with_corrections=False)
    parts = (prompt.PERSONAS[prompt.DEFAULT_PERSONA][1], prompt.CONTRACT, prompt.RULES)
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


def test_the_words_on_whose_pieces_and_which_lists_agree() -> None:
    """The deep prompt review (2026-10-01): the first hint's level line said "every piece ...
    of theirs" (the opponent's) and "which of their pieces makes it" (the player's) in one
    breath; rule 2 kept back the piece that the second hint names; and the lists were said
    complete where the position they faced has none."""
    from sidecoach.coach.coach import IDEA_LEVEL_WORDS

    assert "of the opponent's the sheet names" in IDEA_LEVEL_WORDS
    assert "which of the player's pieces makes it" in IDEA_LEVEL_WORDS
    assert "until the Task says the piece may be named" in prompt.RULES
    assert "complete for the position now" in prompt.CONTRACT
    assert "the position they faced -- says only what its facts state" in prompt.CONTRACT


def test_no_persona_or_an_unknown_one_is_the_default_and_the_rating_sets_the_pitch() -> None:
    """A persona left empty or named wrong (a settings file from another version) is the
    default one, warm and direct; every persona keeps the aim, and with a rating band says
    the level to pitch at (the parity audit, 2026-09-29: pinned before only by the phone's
    replay of the prompts)."""
    default = prompt.persona_text(prompt.DEFAULT_PERSONA)
    assert prompt.DEFAULT_PERSONA == "warm_direct"
    for name in (None, "", "no such persona"):
        assert prompt.persona_text(name) == default, name
        assert prompt.system_message(with_corrections=False, persona=name).startswith(default)
    rating = (
        " The player is rated about 1000 to 1200: pitch what you say at that level -- name "
        "the pattern in plain words, assume the basics, and do not explain what a player of "
        "that strength already knows."
    )
    for name, (_, manner) in prompt.PERSONAS.items():
        plain = prompt.persona_text(name)
        assert manner in plain and rating not in plain, name
        assert prompt.persona_text(name, (1000, 1200)) == plain + rating, name
        others = [m for n, (_, m) in prompt.PERSONAS.items() if n != name]
        assert not any(m in plain for m in others), name


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
    assert text.index(prompt.RULES) < text.index("Lessons from earlier games")
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
    # An example is in the words it asks for: "White is clearly ahead" was neither the
    # message's word nor "you" (the deep prompt review, 2026-10-01).
    assert "clearly ahead" not in prompt.corrections()


def test_the_model_is_never_told_of_messages_it_does_not_get() -> None:
    """The model gets one message, in sections (the contract); "the next message is the
    replacement move", "the review in the last message" and "the messages above" were left
    from the days of many (the deep prompt review, 2026-10-01). Read from the string literals
    (not the docstrings) of the modules that write what the model sees."""
    import ast

    import sidecoach.coach.coach as coach_module
    import sidecoach.coach.conversation as conversation_module
    import sidecoach.coach.engine_tools as tools_module

    stale = re.compile(r"messages above|next message|last message|move's message", re.I)
    found = []
    for module in (coach_module, conversation_module, tools_module, prompt):
        tree = ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8"))
        docs = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docs
                and stale.search(node.value)
            ):
                found.append((module.__name__, node.value[:80]))
    assert not found, found
