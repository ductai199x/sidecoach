"""The voices a person can pick: Kokoro's English ones, American and British, female and
male, and the label each is shown under ("American Female, Sarah").

The list comes from the voices file when it is there (every English voice the model
ships), else from the names Kokoro v1.0 is known to carry, so the picker is never empty
before the first download. Nothing else is offered: the coach speaks English.
"""

from __future__ import annotations

from pathlib import Path

# Kokoro v1.0's English voices, in the order the picker shows them.
ENGLISH_VOICES: tuple[str, ...] = (
    "af_heart",
    "af_sarah",
    "af_bella",
    "af_nicole",
    "af_nova",
    "af_sky",
    "af_alloy",
    "af_aoede",
    "af_jessica",
    "af_kore",
    "af_river",
    "am_adam",
    "am_michael",
    "am_echo",
    "am_liam",
    "am_eric",
    "am_fenrir",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_emma",
    "bf_isabella",
    "bf_alice",
    "bf_lily",
    "bm_george",
    "bm_lewis",
    "bm_daniel",
    "bm_fable",
)
_ACCENT = {"a": "American", "b": "British"}
_SEX = {"f": "Female", "m": "Male"}


def is_english(name: str) -> bool:
    """`af_`, `am_`, `bf_`, `bm_`: American or British, female or male."""
    return len(name) > 3 and name[0] in _ACCENT and name[1] in _SEX and name[2] == "_"


def voice_label(name: str) -> str:
    """ "af_sarah" -> "American Female, Sarah"; a name outside the scheme stays as it is."""
    if not is_english(name):
        return name
    return f"{_ACCENT[name[0]]} {_SEX[name[1]]}, {name[3:].capitalize()}"


def english_voices(voices_path: Path | None = None) -> list[str]:
    """The English voices to offer: the file's, in the picker's order with any the list
    does not know appended, else the known list."""
    known = list(ENGLISH_VOICES)
    if voices_path is None or not voices_path.exists():
        return known
    try:
        import numpy as np

        with np.load(voices_path) as bundle:
            present = {n for n in bundle.files if is_english(n)}
    except Exception:  # noqa: BLE001 - a file that will not open costs nothing but the list
        return known
    if not present:
        return known
    return [v for v in known if v in present] + sorted(present.difference(known))
