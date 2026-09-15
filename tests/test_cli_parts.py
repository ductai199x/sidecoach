"""Small CLI helpers that parse user input."""

from pathlib import Path

from sidecoach.cli import _part_spec


def test_concat_part_spec_keeps_a_drive_letter() -> None:
    assert _part_spec("data/rec") == (Path("data/rec"), None, None)
    assert _part_spec("data/rec:10") == (Path("data/rec"), 10, None)
    assert _part_spec("data/rec:10:20") == (Path("data/rec"), 10, 20)
    assert _part_spec("data/rec::20") == (Path("data/rec"), None, 20)
    assert _part_spec(r"D:\data\rec") == (Path(r"D:\data\rec"), None, None)
    assert _part_spec(r"D:\data\rec:10:20") == (Path(r"D:\data\rec"), 10, 20)
    assert _part_spec(r"D:\data\rec:5") == (Path(r"D:\data\rec"), 5, None)
