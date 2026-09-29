"""Small CLI helpers: parsing user input, and building the coach's parts from the settings."""

from pathlib import Path

from sidecoach.cli import _part_spec, _verifier
from sidecoach.config import Settings


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
