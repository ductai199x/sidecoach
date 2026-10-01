"""The settings over HTTP, for the app's settings window (docs/app-plan.md, section 3).

GET gives the settings, their JSON schema (every field's description, choices and when a
change applies: now, at the next Start, or after a restart -- the `_extra` marks in
config.py), where the file is, which secrets are set, and the audio devices. PUT validates
a whole settings object through the pydantic models, writes the TOML, hands the secrets to
the credential store, applies to the running engine what can be applied, and answers with
what took effect now, what waits for the next Start, and what needs a restart.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ValidationError

from sidecoach import looks
from sidecoach.coach.prompt import PERSONAS
from sidecoach.config import (
    KEY_ENVS,
    Settings,
    save_settings,
    secrets_get,
    secrets_set,
)
from sidecoach.voice.voices import english_voices

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
        "path": str(settings.source),
        "secrets": {
            name: bool(os.environ.get(name) or secrets_get(name))
            for name in _secret_names(settings)
        },
        "keyring": _keyring_available(),
        "key_envs": KEY_ENVS,
        "devices": _audio_devices(),
        "voices": english_voices(settings.voice_model_dir() / "voices-v1.0.bin"),
        "personas": {name: label for name, (label, _) in PERSONAS.items()},
        "looks": looks.catalogue(settings.data_dir),
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


def _merged(base: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
    """`base` with `change` laid over it, section by section: a field not in the change keeps
    its value, and one in it is the change's, whole -- an object too, so a key taken out of
    llm.extra_body goes (the parity audit, 2026-10-01; the phone's SettingsCheck does the
    same)."""
    out = dict(base)
    for key, value in change.items():
        both = isinstance(value, dict) and isinstance(out.get(key), dict)
        out[key] = {**out[key], **value} if both else value
    return out


def _validated(engine: Any, change: dict[str, Any]) -> Settings:
    """The settings as they are now with `change` laid over them, validated; raises
    ValidationError."""
    new = Settings.model_validate(_merged(engine.settings.model_dump(mode="json"), change))
    data_dir = os.environ.get("SIDECOACH_DATA_DIR")
    if data_dir:
        # The shell owns the data directory; the file never overrides it.
        from pathlib import Path

        new.data_dir = Path(data_dir)
    return new


def remember(engine: Any, change: dict[str, Any]) -> None:
    """A choice made on the panel itself -- muted, how the coach listens -- kept in the
    settings file, so the next session starts as this one was left (Tai, 2026-10-01: "mute
    setting isn't remembered between sessions"). The panel applied it already; nothing is
    applied again."""
    new = _validated(engine, change)
    new._source = save_settings(new, engine.settings.source)
    if hasattr(engine, "adopt_settings"):
        engine.adopt_settings(new)
    else:
        engine.settings = new


def update(engine: Any, body: SettingsUpdate) -> dict[str, Any]:
    """Validate, save, store the secrets, apply what can be applied, and say what waits. The
    settings sent are what changed, laid over the settings as they are now: a page opened
    before another saved cannot put back what that one changed (a Settings window left open
    wrote the chess.com username back to empty, Tai 2026-09-30)."""
    try:
        new = _validated(engine, body.settings)
    except ValidationError as exc:
        return {"ok": False, "errors": exc.errors(include_url=False)}
    old = engine.settings
    path = save_settings(new, old.source)
    new._source = path
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

    # What the engine can change on the running session, it does now. A "now" field with
    # nothing running to change is applied all the same: the settings are what the next
    # Start builds from ("At the next Start: Model" was said of a model saved before a
    # game, and read as a restart -- Tai, 2026-09-21). Only a marked "next" field waits.
    if hasattr(engine, "apply_settings"):
        engine.apply_settings(new, changed)
    else:
        engine.settings = new

    def words(names: list[str]) -> str:
        return ", ".join(n.split(".")[-1].replace("_", " ") for n in names)

    engine._emit(
        "sys",
        "info",
        "settings saved"
        + (f"; applied now: {words(now)}" if now else "")
        + (f"; at the next Start: {words(later)}" if later else "")
        + (f"; after a restart: {words(restart)}" if restart else "")
        + ("; nothing changed" if not (now or later or restart) else ""),
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
