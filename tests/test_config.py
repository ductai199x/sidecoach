import os
from pathlib import Path

import pytest

from sidecoach.config import load_dotenv


def test_dotenv_sets_missing_and_keeps_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text('# comment\nNEW_KEY="quoted value"\nEXISTING=from_file\n\nbad line\n')
    monkeypatch.delenv("NEW_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "from_env")

    loaded = load_dotenv(dotenv)

    assert loaded == {"NEW_KEY": "quoted value"}
    assert os.environ["NEW_KEY"] == "quoted value"
    assert os.environ["EXISTING"] == "from_env"
    # A missing file is fine.
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_a_saved_retired_default_moves_on_and_a_chosen_value_stays(tmp_path: Path) -> None:
    """The thresholds became chess.com's (Tai, 2026-10-06): a file that saved the old defaults,
    as every saved file does, reads the new ones; a value the user set is kept."""
    from sidecoach.config import load_settings

    saved = tmp_path / "sidecoach.toml"
    saved.write_text("[coach]\ninaccuracy = 0.04\nmistake = 0.1\nblunder = 0.28\n")
    coach = load_settings(saved).coach
    assert (coach.inaccuracy, coach.mistake, coach.blunder) == (0.05, 0.1, 0.20)
    saved.write_text("[coach]\ninaccuracy = 0.06\nblunder = 0.3\n")
    coach = load_settings(saved).coach
    assert (coach.inaccuracy, coach.blunder) == (0.06, 0.3)
