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
    import os

    assert os.environ["NEW_KEY"] == "quoted value"
    assert os.environ["EXISTING"] == "from_env"


def test_dotenv_missing_file_is_fine(tmp_path: Path) -> None:
    assert load_dotenv(tmp_path / "nope.env") == {}
