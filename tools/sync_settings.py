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
    # The engine a phone starts with: two workers of one thread each and 64 MB where the
    # desktop has three and 256, for the cores and the heat a phone can spare -- the two cores
    # its one engine of two threads took before (the engine audit, 2026-10-04: one thread a
    # search, never several). The engine is built from these, as the desktop's
    # `_make_analyser` builds it, so they are the phone's defaults to change.
    defaults["engine"]["workers"] = 2
    defaults["engine"]["hash_mb"] = 64
    # And depth 14 where the desktop has 18: on Tai's Pixel (2026-09-28) Stockfish 19 does a
    # tenth of the desktop's nodes a second on two threads, depth 18 came after ten seconds
    # and more, never inside the five-second cap, and depth 14 in 2.6 s (3.7 warm). Three
    # threads were no faster, the phone warmer. Tai: "2 threads and depth 14".
    defaults["engine"]["depth"] = 14
    # The captions, the phone's first (Tai, 2026-09-23) and the desktop's since 2026-09-29: the
    # desktop's words say windows and clicks; the phone's, apps, touches and Android's leave to
    # draw over other apps.
    voice = schema["$defs"]["VoiceSettings"]["properties"]
    voice["captions"]["label"] = "Captions over other apps"
    voice["captions"]["description"] = (
        "Shows what the coach says in a small box over other apps, like Live Caption, so you "
        "can read it while you play. Touch and hold the box to move it; double-tap it to see "
        "more; tap it for its buttons; drag it onto the cross at the bottom of the screen to "
        "turn captions off. CC at the top of the panel turns them on and off too. The first "
        "time, Android asks you to let Sidecoach display over other apps."
    )
    voice["captions_keep"]["description"] = (
        "Keeps the caption box on screen all the time, over every app, Sidecoach too: its "
        "buttons, and the coach's last words once it has spoken. Off, the box comes up when the "
        "coach speaks, fades a few seconds after, and steps aside while Sidecoach itself is open. "
        "The pin beside CC at the top of the panel does the same."
    )
    voice["muted"]["description"] = (
        "The coach stays silent; its words still show, on the panel and in the captions. The "
        "speaker button at the top of the panel does the same."
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
    engine["workers"]["default"] = 2
    engine["workers"]["description"] = (
        "How many positions Stockfish works on at once, one processor core each: the position "
        "you face, the best moves in it, and the coach's questions. More answers sooner, and "
        "warms the phone."
    )
    engine["depth"]["default"] = 14
    engine["depth"]["description"] = (
        "How deep Stockfish looks at every position. Deeper is more accurate and slower; 14 is "
        "what a phone reaches in a few seconds, and stronger than any player."
    )
    engine["hash_mb"]["default"] = 64
    engine["hash_mb"]["description"] = (
        "Memory each of those keeps for positions it has already seen, in MB. More helps a deep "
        "search; 64 is plenty on a phone."
    )
    llm = schema["$defs"][refs["llm"]]["properties"]
    llm["provider"]["description"] = (
        "The AI model the coach thinks with. OpenAI is the one the coach was tested on; a model "
        "on your own computer costs nothing per game; with no model the coach still speaks, from "
        "the board and its templates."
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
