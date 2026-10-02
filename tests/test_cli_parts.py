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


class _Behind:
    """An analyser with `pending` verdicts still to come: each poll hands one over."""

    def __init__(self, pending: int, forever: bool = False) -> None:
        self.pending = pending
        self.forever = forever
        self.polls = 0

    def poll(self) -> list[object]:
        self.polls += 1
        if self.pending == 0:
            return []
        if not self.forever:
            self.pending -= 1
        return [f"verdict {self.polls}"]


def test_the_end_of_a_game_waits_for_the_engine_to_catch_up_and_no_longer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`_Assessing.wait` at the end of a game: every verdict still to come is drained, polled
    every 50 ms while the engine is behind; an engine that never catches up is let go at the
    timeout with what it gave. On a clock the test owns (the replay and record commands use
    it; the analysis tests wait on the analyser's own signals instead)."""
    from tests.fakes import ManualClock

    clock = ManualClock()
    monkeypatch.setattr(cli, "time", clock.time_module())
    assessing = cli._Assessing(Settings(data_dir=tmp_path), enabled=False)
    assert assessing.wait() == [], "no engine: nothing to wait for"
    behind = _Behind(pending=3)
    assessing.analyser = behind  # type: ignore[assignment]
    assert assessing.wait(timeout=30.0) == ["verdict 1", "verdict 2", "verdict 3"]
    assert clock.sleeps == [0.05, 0.05, 0.05], "polled each 50 ms while behind, then not"
    clock.sleeps.clear()
    stuck = _Behind(pending=1, forever=True)
    assessing.analyser = stuck  # type: ignore[assignment]
    start = clock.now
    drained = assessing.wait(timeout=1.0)
    waited = clock.now - start
    assert 1.0 <= waited < 1.0 + 0.05 + 1e-9, f"let go at the first poll past 1 s: {waited}"
    assert set(clock.sleeps) == {0.05}
    polls = len(clock.sleeps) + 1  # one a sleep, and the last drain on the way out
    assert drained == [f"verdict {n}" for n in range(1, polls + 1)], "with every verdict it gave"
