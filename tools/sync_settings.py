"""Ship the desktop's settings schema and defaults to the app.

The panel's settings page is driven by the JSON schema pydantic generates from
`sidecoach/config.py` -- every field's description, its choices, whether a change applies
now, at the next Start or after a restart. The phone runs that same page, so it needs that
same schema; it is a description of the settings, not a value, so it ships as an asset and
is refreshed here whenever `config.py` changes.

The defaults come with it, as the settings a phone starts from. What the phone actually
honours is a subset -- there is no voice yet, no CUDA pack, no tools directory -- and the
page shows every field either way, which is the desktop's behaviour for a setting that does
not apply to the machine in front of you.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/sync_settings.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

from sidecoach.config import KEY_ENVS, Settings  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets" / "settings"


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        schema = Settings.model_json_schema()
    defaults = Settings().model_dump(mode="json")
    # The phone keeps its own folders; the page shows them read-only.
    defaults["data_dir"] = ""
    defaults["tools_dir"] = ""
    # The engine a phone starts with: two threads and 64 MB where the desktop has four and
    # 256, for the cores and the heat a phone can spare. The engine is built from these, as
    # the desktop's `_make_analyser` builds it, so they are the phone's defaults to change.
    defaults["engine"]["threads"] = 2
    defaults["engine"]["hash_mb"] = 64
    # The phone's own setting: the coach's words in a box over other apps, the way Live Caption
    # shows speech (Tai, 2026-09-23). The desktop has no such box, so the field is added here,
    # to the phone's schema alone, and the desktop's page never shows it.
    voice = schema["$defs"]["VoiceSettings"]["properties"]
    voice["captions"] = {
        "default": False,
        "description": "Shows what the coach says in a small box over other apps, like Live "
        "Caption, so you can read it while you play. Touch and hold the box to move it; "
        "double-tap it to see more; tap it for its buttons; drag it onto the cross at the "
        "bottom of the screen to turn captions off. The first time, Android asks you to let "
        "Sidecoach display over other apps.",
        "title": "Captions",
        "type": "boolean",
        "applies": "now",
        "label": "Captions over other apps",
        "section": "The coach's voice",
    }
    defaults["voice"]["captions"] = False
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (OUT / "defaults.json").write_text(json.dumps(defaults), encoding="utf-8")
    (OUT / "key_envs.json").write_text(json.dumps(KEY_ENVS), encoding="utf-8")
    print(
        f"schema {len(json.dumps(schema)) / 1e3:.0f} kB, defaults "
        f"{len(json.dumps(defaults)) / 1e3:.1f} kB -> {OUT}"
    )


if __name__ == "__main__":
    main()
