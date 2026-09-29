"""Small CLI helpers: parsing user input, and building the coach's parts from the settings."""

from pathlib import Path

import pytest

from sidecoach import cli
from sidecoach.analysis.computed import Computed, ComputedFirst
from sidecoach.cli import _part_spec, _verifier
from sidecoach.config import LLMSettings, Settings


def test_the_coachs_engine_reads_the_analysers_store_and_closes_with_the_coach(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The coach's engine answers from what the analyser computed and starts its Stockfish
    only at a miss (Tai, 2026-09-29); the coach closed, it is closed too, where it used to
    run on until the app quit."""
    monkeypatch.setattr(cli, "find_stockfish", lambda _settings: tmp_path / "stockfish")
    computed = Computed()
    coaching = cli._Coaching(Settings(llm=LLMSettings(provider="none")), True, computed)
    engine = coaching.coach.engine if coaching.coach is not None else None
    assert isinstance(engine, ComputedFirst) and engine.computed is computed
    assert engine.engine._engine is None, "no Stockfish started before a search"  # type: ignore[attr-defined]
    coaching.close()
    assert engine._closed


def test_concat_part_spec_keeps_a_drive_letter() -> None:
    assert _part_spec("data/rec") == (Path("data/rec"), None, None)
    assert _part_spec("data/rec:10") == (Path("data/rec"), 10, None)
    assert _part_spec("data/rec:10:20") == (Path("data/rec"), 10, 20)
    assert _part_spec("data/rec::20") == (Path("data/rec"), None, 20)
    assert _part_spec(r"D:\data\rec") == (Path(r"D:\data\rec"), None, None)
    assert _part_spec(r"D:\data\rec:10:20") == (Path(r"D:\data\rec"), 10, 20)
    assert _part_spec(r"D:\data\rec:5") == (Path(r"D:\data\rec"), 5, None)


def test_a_verifier_that_cannot_be_built_leaves_the_coach_without_one() -> None:
    """A name mistyped, or a provider with no key: the coach runs without a second opinion,
    where raising stopped it from starting (the parity audit, 2026-09-29)."""
    settings = Settings()
    provider = object()
    for kind, expected in (("", None), ("same", provider), ("opneai", None)):
        settings.coach.verify = kind
        assert _verifier(settings, provider) is expected, kind
