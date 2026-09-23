"""Record what the desktop's voice front end makes of a corpus, for VoiceParityTest.

The coach's lines become sound through a chain the phone ports link by link: the notation read
out in words (`sidecoach.voice.notation.speak_notation`), the text phonemized by espeak-ng
through phonemizer (`phonemizer.phonemize`, the way kokoro-onnx calls it), the phonemes kept
to Kokoro's vocabulary and tokenized, cut into batches (`chunker.split_phonemes`) with the
pause each calls for, synthesised, and the silence trimmed (`trim.trim`). This records, from
the desktop's own code:

- for each text of the corpus: the spoken form, phonemizer's raw output, Kokoro's phonemes, the
  token ids, the batches and their pauses -- of the text as given and of its spoken form;
- the trim of a few synthetic signals;
- the batches of a few long phoneme strings (sentence, clause, word and mid-word cuts);
- the pieces the phone speaks each spoken form in (`phone_speech.plan`, the phone's own plan:
  the desktop speaks whole lines), at the coach's speed and at a slower and a faster one;
- Kokoro's audio for a few short lines, full precision as the phone runs it (the phone's must
  match within a tolerance: the two runtimes' kernels differ in their last bits; int8 differed
  in more than that on the Pixel's ARM cores, and was slower there).

The corpus: every coach message in the recorded coach sessions, and cases written for the
edges -- punctuation at the start and the end, a line of marks alone, decimals, quotes,
brackets, an ellipsis, a dash, names with accents, contractions, numbers, notation, a line
break, and texts long enough to need several batches.

phonemizer splits and joins lines on `os.linesep`; the phone is Linux, so this runs with it
set to "\\n" whatever the desktop's own is.

    uv run --no-sync python D:/sidecoach-android/tools/make_voice_fixtures.py   (desktop repo)
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

os.linesep = "\n"  # the phone's; see above

import phonemizer  # noqa: E402
from kokoro_onnx import Kokoro  # noqa: E402
from kokoro_onnx.chunker import pause_after, split_phonemes  # noqa: E402
from kokoro_onnx.tokenizer import Tokenizer  # noqa: E402
from kokoro_onnx.trim import trim  # noqa: E402

from sidecoach.voice.notation import speak_notation  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import phone_speech  # noqa: E402

ANDROID = Path("D:/sidecoach-android")
OUT = ANDROID / "app" / "src" / "androidTest" / "assets" / "voice" / "voice.json"
SESSION = ANDROID / "app" / "src" / "androidTest" / "assets" / "chess" / "session.json"
MODEL_DIR = Path("D:/chess-coaching-companion/data/models/kokoro")

EDGES = [
    "Nf3, still theory.",
    "Qxf7# -- that is mate!",
    "The bishop on c4 is loose... what now?",
    "You are 1.5 pawns up; that is 19,99 on the clock.",
    "(Note: castle kingside.)",
    "“Quoted” words «here» and ¡hola!",
    "—",
    "...",
    "?!",
    "Hello, world",
    ", leading comma",
    "Trailing marks!!!",
    "A dash — in the middle — twice.",
    "Kostić played Bxf7+ and won; don't miss it, it's the player's trick.",
    "Accuracy 94.2. Your 15 moves: 1 best, 14 book.",
    "You played white and won by checkmate in 8 moves. The opening was the C50 Italian Game: Blackburne-Kostić Gambit.",
    "O-O-O, then e8=Q+ and 1-0.",
    "First line.\nSecond line, after a break.",
    "   spaces   around   ",
    "[bracketed] {braced} (parenthesised)",
    "Is it 3.5 or 3, 5?",
    "exd5 e.p. is not a thing here; cxd5 is.",
    "What were you expecting after Nxe5?",
    "Rf1 -- very good.",
    "12... Bg5 13. Nxg5 O-O",
]

LONG = [
    " ".join(["The knight on f3 guards the pawn on e5, and the bishop on c4 eyes f7."] * 12),
    " ".join(["No stop at all here just words going on and on without a break"] * 12),
]


def corpus() -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    data = json.loads(SESSION.read_text(encoding="utf-8"))
    for run in data["runs"]:
        for step in run["steps"]:
            for message in step.get("messages", []):
                text = message.get("text", "")
                if text and text not in seen:
                    seen.add(text)
                    texts.append(text)
    for text in EDGES + LONG:
        if text not in seen:
            seen.add(text)
            texts.append(text)
    return texts


def front(tokenizer: Tokenizer, text: str) -> dict:
    raw = phonemizer.phonemize(text, "en-us", preserve_punctuation=True, with_stress=True)
    phonemes = tokenizer.phonemize(text, "en-us")
    collapsed = " ".join(phonemes.split())
    batches = split_phonemes(collapsed)
    return {
        "raw": raw,
        "phonemes": phonemes,
        "tokens": tokenizer.tokenize(phonemes, limit=None),
        "batches": batches,
        "pauses": [pause_after(b, 0.25, 0.1) for b in batches[:-1]] + ([0.0] if batches else []),
    }


def plans(tokenizer: Tokenizer, cases: list[dict]) -> list[dict]:
    out = []
    for case in cases:
        for speed in (1.0, 0.7, 1.4):
            pieces = phone_speech.plan(tokenizer, case["spoken"], speed)
            out.append({"text": case["spoken"], "speed": speed, "pieces": [p for p, _ in pieces], "pauses": [z for _, z in pieces]})
    return out


def signals() -> list[dict]:
    rng = np.random.default_rng(7)
    out = []
    t = np.arange(24000, dtype=np.float32) / 24000
    shapes = [
        np.concatenate([np.zeros(5000), 0.5 * np.sin(2 * np.pi * 220 * t[:10000]), np.zeros(7000)]),
        np.concatenate([1e-5 * rng.standard_normal(3000), 0.3 * rng.standard_normal(8000), 1e-5 * rng.standard_normal(9000)]),
        0.2 * np.sin(2 * np.pi * 440 * t[:6000]),
        np.zeros(4000),
        np.concatenate([np.zeros(600), 0.9 * np.ones(300), np.zeros(600)]),
        np.concatenate([0.001 * np.sin(2 * np.pi * 100 * t[:9000]), 0.4 * np.sin(2 * np.pi * 300 * t[:9000]), 0.0005 * rng.standard_normal(9000)]),
    ]
    for y in shapes:
        y = y.astype(np.float32)
        _, (start, end) = trim(y)
        out.append({"y": base64.b64encode(y.tobytes()).decode(), "start": int(start), "end": int(end)})
    return out


def long_strings(tokenizer: Tokenizer) -> list[dict]:
    random.seed(3)
    strings = [
        " ".join(tokenizer.phonemize(LONG[0]).split()),
        " ".join(tokenizer.phonemize(LONG[1]).split()),
        "ə" * 1200,
        ("ab " * 300).strip(),
        "sentence one. " * 60 + "clause, " * 40,
    ]
    return [{"phonemes": s, "batches": split_phonemes(s)} for s in strings]


def audio(kokoro: Kokoro) -> list[dict]:
    out = []
    for text in ["Nf3, still theory.", "Good. What does your knight want?", "Ready."]:
        spoken = speak_notation(text).strip()
        samples, rate = kokoro.create(spoken, voice="af_sarah", speed=1.0, lang="en-us")
        samples = np.asarray(samples, dtype=np.float32)
        out.append({
            "text": spoken,
            "rate": rate,
            "samples": base64.b64encode(samples.tobytes()).decode(),
        })
    return out


def main() -> None:
    tokenizer = Tokenizer()
    texts = corpus()
    cases = []
    for text in texts:
        spoken = speak_notation(text).strip()
        cases.append({"text": text, "spoken": spoken, "given": front(tokenizer, text), "said": front(tokenizer, spoken)})
    fixtures = {
        "texts": cases,
        "trims": signals(),
        "long": long_strings(tokenizer),
        "plans": plans(tokenizer, cases),
    }
    # The model the phone runs: the full-precision one (Tai, 2026-09-23).
    model = MODEL_DIR / "kokoro-v1.0.onnx"
    if model.exists():
        # On the CPU, as the phone runs it: kokoro-onnx would take the desktop's GPU.
        import onnxruntime as rt

        session = rt.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        kokoro = Kokoro.from_session(session, str(MODEL_DIR / "voices-v1.0.bin"))
        fixtures["audio"] = audio(kokoro)
    else:
        print(f"no {model}: no audio fixtures")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixtures, ensure_ascii=False), encoding="utf-8")
    print(f"{len(cases)} texts, {len(fixtures['trims'])} trims, {len(fixtures['long'])} long strings,"
          f" {len(fixtures['plans'])} plans ({sum(len(x['pieces']) > 1 for x in fixtures['plans'])} in pieces),"
          f" {len(fixtures.get('audio', []))} audio -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
