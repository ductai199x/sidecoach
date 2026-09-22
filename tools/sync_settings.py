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
