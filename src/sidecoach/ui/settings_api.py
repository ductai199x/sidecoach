"""The settings over HTTP, for the app's settings window (docs/app-plan.md, section 3).

GET gives the settings, their JSON schema (every field's description, choices and when a
change applies: now, at the next Start, or after a restart -- the `_extra` marks in
config.py), where the file is, which secrets are set, and the audio devices. PUT validates
a whole settings object through the pydantic models, writes the TOML, hands the secrets to
the credential store, applies to the running engine what can be applied, and answers with
what took effect now, what waits for the next Start, and what needs a restart.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from sidecoach.config import (
    KEY_ENVS,
    Settings,
    save_settings,
    secrets_get,
    secrets_set,
    settings_path,
)

SECRET_NAMES: tuple[str, ...] = ("OPENAI_API_KEY", "DASHSCOPE_API_KEY")


class SettingsUpdate(BaseModel):
    settings: dict[str, Any]
    secrets: dict[str, str | None] = {}


def _audio_devices() -> dict[str, list[dict[str, Any]]]:
    """The input and output devices sounddevice sees, or empty lists without it."""
    try:
        import sounddevice as sd  # type: ignore[import-not-found]

        devices = sd.query_devices()
        inputs = [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d.get("max_input_channels", 0) > 0
        ]
        outputs = [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d.get("max_output_channels", 0) > 0
        ]
        return {"inputs": inputs, "outputs": outputs}
    except Exception:  # noqa: BLE001 - no audio stack is a fact, not an error here
        return {"inputs": [], "outputs": []}


def _secret_names(settings: Settings) -> list[str]:
    names = list(SECRET_NAMES)
    custom = settings.llm.api_key_env
    if custom and custom not in names:
        names.append(custom)
    return names


def _keyring_available() -> bool:
    try:
        import keyring  # type: ignore[import-not-found]  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def describe(settings: Settings) -> dict[str, Any]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        schema = Settings.model_json_schema()
    return {
        "settings": settings.model_dump(mode="json"),
        "schema": schema,
        "path": str(settings_path()),
        "secrets": {
            name: bool(os.environ.get(name) or secrets_get(name))
            for name in _secret_names(settings)
        },
        "keyring": _keyring_available(),
        "key_envs": KEY_ENVS,
        "devices": _audio_devices(),
    }


def _applies(schema: dict[str, Any]) -> dict[str, str]:
    """'section.field' -> now | next | restart, from the schema's marks."""
    out: dict[str, str] = {}
    defs = schema.get("$defs", {})
    for name, prop in schema.get("properties", {}).items():
        if "$ref" in prop:
            model = defs.get(prop["$ref"].rsplit("/", 1)[-1], {})
            for field, fprop in model.get("properties", {}).items():
                out[f"{name}.{field}"] = fprop.get("applies", "next")
        else:
            out[name] = prop.get("applies", "next")
    return out


def _changed(old: dict[str, Any], new: dict[str, Any], prefix: str = "") -> list[str]:
    changed: list[str] = []
    for key in set(old) | set(new):
        a, b = old.get(key), new.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changed.extend(_changed(a, b, f"{prefix}{key}."))
        elif a != b:
            changed.append(f"{prefix}{key}")
    return sorted(changed)


def update(engine: Any, body: SettingsUpdate) -> dict[str, Any]:
    """Validate, save, store the secrets, apply what can be applied, and say what waits."""
    try:
        new = Settings.model_validate(body.settings)
    except ValidationError as exc:
        return {"ok": False, "errors": exc.errors(include_url=False)}
    data_dir = os.environ.get("SIDECOACH_DATA_DIR")
    if data_dir:
        # The shell owns the data directory; the file never overrides it.
        from pathlib import Path

        new.data_dir = Path(data_dir)
    old = engine.settings
    path = save_settings(new)
    changed = _changed(old.model_dump(mode="json"), new.model_dump(mode="json"))
    applies = _applies(Settings.model_json_schema())
    now = [c for c in changed if applies.get(c) == "now"]
    later = [c for c in changed if applies.get(c) == "next"]
    restart = [c for c in changed if applies.get(c) == "restart"]

    stored: dict[str, bool] = {}
    for name, value in body.secrets.items():
        if name not in _secret_names(new):
            continue
        stored[name] = secrets_set(name, value)
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)

    engine.settings = new
    with contextlib.suppress(Exception):
        if "voice.voice" in changed:
            engine.set_voice(new.voice.voice)
        if "voice.listen" in changed:
            engine.set_listen(new.voice.listen)
    engine._emit(
        "sys",
        "info",
        "settings saved"
        + (f"; applied now: {', '.join(now)}" if now else "")
        + (f"; at the next start: {', '.join(later)}" if later else "")
        + (f"; after a restart: {', '.join(restart)}" if restart else ""),
    )
    return {
        "ok": True,
        "path": str(path),
        "changed": changed,
        "now": now,
        "next": later,
        "restart": restart,
        "secrets": stored,
    }
