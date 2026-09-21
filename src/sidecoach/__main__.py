"""`python -m sidecoach`: the CLI, for the embedded interpreter the desktop shell runs
(`resources/python/python.exe -m sidecoach panel ...`), where no console script exists."""

from sidecoach.cli import app

app()
