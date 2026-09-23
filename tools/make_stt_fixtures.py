"""Record what the desktop's listening makes of speech, for SttParityTest.

What the player says becomes an ask through a chain the phone ports link by link: Silero's speech
probability per 32 ms frame (`voice.vad.SileroGate`) and the endpointer on top of it
(`Endpointer`), Parakeet's transcript (onnx-asr's NeMo TDT path, on the model the phone runs --
Parakeet TDT-CTC 110M int8 -- on the CPU), and the chess correction against the legal moves (`voice.correct`). This records,
from the desktop's own code:

- speech clips of things a player says, spoken by Kokoro and brought to 16 kHz: their samples,
  Parakeet int8's text and token ids, Silero's probability for every frame, and the endpointer's
  events on those probabilities;
- `correct` and `normalise` on a corpus of transcripts -- the recogniser's typical slips as well
  as clean ones -- against a few boards, and difflib's ratio on pairs of move forms.

    uv run --no-sync python D:/sidecoach-android/tools/make_stt_fixtures.py   (desktop repo)
"""

from __future__ import annotations

import base64
import difflib
import json
from pathlib import Path

import chess
import numpy as np
import onnxruntime as rt

from sidecoach.voice.correct import correct, normalise, spoken_forms
from sidecoach.voice.vad import HOP, Endpointer, SileroGate

OUT = Path("D:/sidecoach-android/app/src/androidTest/assets/voice/stt.json")
KOKORO = Path("D:/chess-coaching-companion/data/models/kokoro")

SAID = [
    "Knight to f three.",
    "What should I play here?",
    "Is my bishop on c four safe?",
    "e takes d five",
    "Castle kingside.",
    "Hint, please.",
    "Why was that bad?",
    "Queen takes f seven, check.",
]

TRANSCRIPTS = [
    "Night to F3.",
    "nine to f3",
    "Knight F-tree",
    "E-Tix D5",
    "e x d five",
    "bishop takes see six",
    "rock to d1",
    "Roka 1",
    "castle short",
    "castle long",
    "king side castle",
    "the upon pushes",
    "What should I play here?",
    "Is my knight on f3 safe?",
    "queen takes f7 czech",
    "pawn to e four",
    "e for",
    "bishop to b five",
    "knight takes the pawn on e five",
    "play knight to f three here",
    "Hmm, I think the rook to e1 is good.",
    "what about bee five",
    "age takes g six",
    "queen to h five",
    "knight c three",
    "knight to see three",
    "",
    "   ",
    "O-O",
    "Nf3",
    # Promotions, and the 110M's own ways of hearing things.
    "E8 Queen",
    "e eight, queen",
    "e eight equals queen",
    "e eight knight",
    "pawn to e eight",
    "e takes d eight queen",
    "He takes D five.",
    "He ate queen",
]

BOARDS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 1 5",
    "3r4/4P1k1/8/8/8/8/5K2/8 w - - 0 1",
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
]


def speech(text: str) -> np.ndarray:
    """Kokoro's voice for the text, at 16 kHz (linear resampling: any clip will do, the phone
    reads the same samples)."""
    from kokoro_onnx import Kokoro

    from sidecoach.voice.notation import speak_notation

    global _kokoro
    if _kokoro is None:
        session = rt.InferenceSession(str(KOKORO / "kokoro-v1.0.onnx"), providers=["CPUExecutionProvider"])
        _kokoro = Kokoro.from_session(session, str(KOKORO / "voices-v1.0.bin"))
    samples, rate = _kokoro.create(speak_notation(text), voice="am_michael", speed=1.0, lang="en-us")
    samples = np.asarray(samples, dtype=np.float32)
    n = int(len(samples) * 16000 / rate)
    at = np.linspace(0, len(samples) - 1, n)
    out = np.interp(at, np.arange(len(samples)), samples).astype(np.float32)
    # Half a second of quiet either side, as a person leaves when they speak.
    pad = np.zeros(8000, dtype=np.float32)
    return np.concatenate([pad, out, pad])


_kokoro = None


def main() -> None:
    import onnx_asr

    # The phone's recogniser: Parakeet TDT-CTC 110M int8 (the desktop's own runs the 0.6B v3).
    asr = onnx_asr.load_model("OpenVoiceOS/nvidia-parakeet-tdt_ctc-110m-onnx", quantization="int8", providers=["CPUExecutionProvider"])
    timed = asr.with_timestamps()
    gate = SileroGate()
    assert gate.load()
    clips = []
    for text in SAID:
        audio = speech(text)
        result = timed.recognize(audio, sample_rate=16000)
        gate.reset()
        probs = []
        for start in range(0, len(audio), HOP):
            probs.append(gate.probability(audio[start : start + HOP]))
        endpointer = Endpointer()
        events = [endpointer.feed(p) for p in probs]
        clips.append({
            "said": text,
            "samples": base64.b64encode(audio.tobytes()).decode(),
            "text": (asr.recognize(audio, sample_rate=16000) or "").strip(),
            "tokens": result.tokens,
            "probs": probs,
            "events": events,
        })
        print(f"{text!r} -> {clips[-1]['text']!r}")

    corrections = []
    for fen in BOARDS:
        board = chess.Board(fen)
        for raw in TRANSCRIPTS:
            fixed = correct(raw, board)
            corrections.append({"fen": fen, "raw": raw, "text": fixed.text, "moves": fixed.moves})
    for raw in TRANSCRIPTS:
        fixed = correct(raw, None)
        corrections.append({"fen": None, "raw": raw, "text": fixed.text, "moves": fixed.moves})
    normalised = [{"raw": raw, "norm": normalise(raw)} for raw in TRANSCRIPTS]

    forms = list(spoken_forms(chess.Board(BOARDS[1])).keys())
    ratios = []
    for phrase in ["knight to f three", "night to f tree", "bishop takes c six", "castle king side", "e takes d five", "pawn e four"]:
        for form in forms:
            ratios.append({"a": phrase, "b": form, "ratio": difflib.SequenceMatcher(None, phrase, form).ratio()})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "clips": clips,
        "corrections": corrections,
        "normalised": normalised,
        "ratios": ratios,
    }), encoding="utf-8")
    print(f"{len(clips)} clips, {len(corrections)} corrections, {len(ratios)} ratios -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
