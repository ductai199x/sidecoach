"""Ship the desktop's settings schema and defaults to the app.

The panel's settings page is driven by the JSON schema pydantic generates from
`sidecoach/config.py` -- every field's description, its choices, whether a change applies
now, at the next Start or after a restart. The phone runs that same page, so it needs that
same schema; it is a description of the settings, not a value, so it ships as an asset and
is refreshed here whenever `config.py` changes.

The defaults come with it, as the settings a phone starts from. The schema is trimmed to
what the phone honours -- its folders, devices, GPU switches and the rest the desktop has are
left out -- so the page never shows an option that does nothing on the phone.

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
        "section": "Captions",
    }
    # Whether the box stays up between lines (Tai, 2026-09-24: "the caption box is always
    # on screen if true and if false then fade away after some time").
    voice["captions_keep"] = {
        "default": False,
        "description": "Keeps the caption box on screen all the time, with the coach's last "
        "words in it. Off, the box fades a few seconds after the coach stops talking and comes "
        "back when it speaks again. Either way it steps aside while Sidecoach itself is open.",
        "title": "Captions Keep",
        "type": "boolean",
        "applies": "now",
        "label": "Keep captions on screen",
        "section": "Captions",
    }
    # How the box looks (Tai, 2026-09-23: "the user can tune things like text color, bg
    # color, opacity, and text size").
    voice["captions_text_color"] = {
        "default": "#FFFFFF",
        "description": "The colour of the captions' words.",
        "title": "Captions Text Color",
        "type": "string",
        "applies": "now",
        "kind": "color",
        "label": "Text colour",
        "section": "Captions",
    }
    voice["captions_background"] = {
        "default": "#1F1F1F",
        "description": "The colour of the box behind the words.",
        "title": "Captions Background",
        "type": "string",
        "applies": "now",
        "kind": "color",
        "label": "Background colour",
        "section": "Captions",
    }
    voice["captions_opacity"] = {
        "default": 92,
        "minimum": 20,
        "maximum": 100,
        "description": "How solid the box is, in percent: 100 hides what is behind it, lower "
        "lets it show through.",
        "title": "Captions Opacity",
        "type": "integer",
        "applies": "now",
        "label": "Background opacity (%)",
        "section": "Captions",
    }
    voice["captions_text_size"] = {
        "default": 17,
        "minimum": 12,
        "maximum": 32,
        "description": "How big the captions' words are; 17 is the usual size.",
        "title": "Captions Text Size",
        "type": "integer",
        "applies": "now",
        "label": "Text size",
        "section": "Captions",
    }
    defaults["voice"].update(
        captions=False,
        captions_keep=False,
        captions_text_color="#FFFFFF",
        captions_background="#1F1F1F",
        captions_opacity=92,
        captions_text_size=17,
    )
    # What the phone does not honour is not on its page (Tai's rule of 2026-09-21: no dead
    # option shown; 2026-09-24: the phone's page showed the desktop's devices, folders, GPU
    # and capture switches, none of which the phone reads). The folders are the phone's own;
    # the devices are the phone's one microphone and speaker; the voice and the recogniser run
    # on the CPU; the engine is the bundled one; nothing is recorded.
    hidden = {
        "capture": ["backend", "record"],
        "engine": ["stockfish_path"],
        "voice": ["model_dir", "tts_device", "stt_device", "mic_device", "output_device", "ptt_key"],
    }
    refs = {name: prop["$ref"].split("/")[-1] for name, prop in schema["properties"].items() if "$ref" in prop}
    for section, keys in hidden.items():
        props = schema["$defs"][refs[section]]["properties"]
        for key in keys:
            del props[key]
    for name in ("data_dir", "tools_dir", "gpu"):
        del schema["properties"][name]
    # Development fields are never shown in the app (Tai: no dev, dead or unwired option on
    # the page; `dev` in config.py's marks). The desktop's page hides them only in its shell,
    # and the phone's page is not the shell, so they go from the phone's schema. Their values
    # stay in the defaults, which is what the code reads (an empty `llm.extra_body`, the
    # presets' own key names and reasoning switches).
    for model in schema["$defs"].values():
        props = model.get("properties", {})
        for key in [k for k, v in props.items() if v.get("dev")]:
            del props[key]
    # The phone reads the frame rate when a session starts.
    schema["$defs"][refs["capture"]]["properties"]["fps"]["applies"] = "next"
    # What the desktop's words say of a computer, said of the phone (plain, and true there):
    # the engine's phone-sized defaults, a model on the person's own computer rather than
    # "this" one, and the talk button where the desktop has a key held from any window.
    engine = schema["$defs"][refs["engine"]]["properties"]
    engine["threads"]["default"] = 2
    engine["threads"]["description"] = (
        "Processor threads Stockfish may use. More is faster to a point, and warms the phone; "
        "leave some for the game."
    )
    engine["hash_mb"]["default"] = 64
    engine["hash_mb"]["description"] = (
        "Memory Stockfish keeps for positions it has already seen, in MB. More helps a deep "
        "search; 64 is plenty on a phone."
    )
    llm = schema["$defs"][refs["llm"]]["properties"]
    llm["provider"]["description"] = (
        "Who the coach thinks with. OpenAI is what the coach was evaluated on; a model on your "
        "own computer costs nothing per game; with no model the coach still speaks, from the "
        "board and its templates."
    )
    llm["provider"]["choice_labels"]["local"] = "A model on your own computer (LM Studio, Ollama, vLLM)"
    llm["base_url"]["description"] = (
        "Where the model is served. For a model on your computer, that computer's address on "
        "your network and LM Studio's port, such as http://192.168.1.20:1234/v1. Empty means "
        "http://localhost:1234/v1, which on a phone is the phone itself."
    )
    listen = schema["$defs"][refs["voice"]]["properties"]["listen"]
    listen["description"] = (
        "How the coach hears you. To talk while you hold, press and hold the round button "
        "beside the question box."
    )
    listen["choice_labels"]["ptt"] = "While I hold the talk button"
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
