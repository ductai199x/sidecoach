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
  as clean ones -- against a few boards, every case of tests/test_voice_correct.py on its own
  board, and the characters where Python's `\\w` and `\\s` are not ICU's; every spoken form of
  every legal move on those boards; difflib's ratio on pairs of move forms; and the endpointer
  on scripted probabilities with its thresholds changed, the coach speaking and not.

    uv run --no-sync python android/tools/make_stt_fixtures.py   (desktop repo)
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

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "voice" / "stt.json"
KOKORO = (Path(__file__).resolve().parents[2] / "data/models/kokoro")

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

# tests/test_voice_correct.py's, every one, on the board the test reads it on (normalise's
# only normalised): after 1.e4 f5 2.exf5 g6, the promotion, the Scandinavian, the start.
AFTER_G6 = "rnbqkbnr/ppppp2p/6p1/5P2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
PROMOTION = "8/4P1k1/8/8/8/8/5K2/8 w - - 0 1"
SCANDINAVIAN = BOARDS[5]
CASTLED = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
NORMALISED = [
    "Night F3",
    "Can I take on D4 with the 9?",
    "The upon is passed, should I push it?",
    "Rooka 1 to D1.",
    "Roka 1 to d1",
    "Why was E-Tix D5 a mistake?",
    "Why was eTakes D5 a mistake?",
    "Why was Exd Five a mistake",
    "Queen H Five Czech",
    "Castle King's side.",
    "should I push the pawn to e for",
    "bishop see six",
]
CORRECTED = [
    ("Night F3", chess.STARTING_FEN),
    ("should I play knight to f three here", chess.STARTING_FEN),
    ("why was E-Tix D5 a mistake", AFTER_G6),
    ("is f takes g six good", AFTER_G6),
    ("what about bishop d three or knight f three", AFTER_G6),
    ("hint", chess.STARTING_FEN),
    ("what should I play here", chess.STARTING_FEN),
    ("the a pawn is passed, should I push it", chess.STARTING_FEN),
    *[(heard, PROMOTION) for heard in ("E8 Queen", "e eight, queen", "e eight equals queen",
                                        "e eight promotes to queen", "pawn to e eight", "e eight",
                                        "e eight knight", "e eight promotes to rook")],
    ("Night to F3", PROMOTION),
    ("pawn to e four", SCANDINAVIAN),
    ("is the pawn on d five weak", SCANDINAVIAN),
    ("knight takes the pawn on e five", SCANDINAVIAN),
    ("is my knight on f three safe", chess.STARTING_FEN),
    ("knight takes f three", chess.STARTING_FEN),
    ("e takes d five", SCANDINAVIAN),
    ("e takes d five", AFTER_G6),
    ("Night F3 Czech", None),
]
# Where Python's \w and \s are not ICU's: "½" and "²" are word characters to Python, a combining
# mark is not; a character past the BMP is one; the separators Python counts as space. On the
# start and on no board.
UNICODE = [
    "it ended ½-½, a draw",
    "e4² or e⁴",
    "café knight f three",
    "knight \U0001F600 f three",
    "knight f three\x1ctakes\x0be\x85four",
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
    pairs = CORRECTED + [(raw, fen) for raw in UNICODE for fen in (chess.STARTING_FEN, None)]
    for raw, fen in pairs:
        fixed = correct(raw, None if fen is None else chess.Board(fen))
        corrections.append({"fen": fen, "raw": raw, "text": fixed.text, "moves": fixed.moves})
    normalised = [{"raw": raw, "norm": normalise(raw)} for raw in TRANSCRIPTS + NORMALISED + UNICODE]

    # The endpointer on scripted probabilities, with the desktop test's own thresholds and
    # others, the coach speaking and not: its event and whether it is inside an utterance.
    endpointer = []
    rng = np.random.default_rng(11)
    walk = [float(round(p, 3)) for p in rng.random(300)]
    coach = [bool(c) for c in rng.random(300) < 0.3]
    for kwargs, feeds in (
        ({"end_ms": 96, "min_utterance_ms": 64}, [(p, False) for p in [0.9, 0.9] + [0.8] * 5 + [0.1] * 3]),
        ({"end_ms": 64, "min_utterance_ms": 320}, [(p, False) for p in [0.9, 0.9, 0.1, 0.1]]),
        ({}, [(0.6, True), (0.6, True), (0.9, True), (0.9, True)]),
        ({}, [(0.85, True), (0.85, True), (0.5, False), (0.35, False), (0.34, False)] + [(0.1, False)] * 13),
        ({}, list(zip(walk, coach))),
        ({"start": 0.7, "end": 0.2, "start_frames": 3, "end_ms": 200, "start_while_speaking": 0.95,
          "min_utterance_ms": 100, "frame_ms": 20}, list(zip(walk, coach))),
    ):
        ep = Endpointer(**kwargs)
        events = []
        for p, speaking in feeds:
            events.append([ep.feed(p, coach_speaking=speaking), ep.speaking])
        endpointer.append({"args": kwargs, "feeds": [[p, s] for p, s in feeds], "events": events})

    # Every spoken form of every legal move, and the move it is, board by board, in the order
    # they are made: since desktop 1e2e10f a move's forms are an ordered dict, the phone's
    # order, where they were a set whose order the string hash seed decided.
    all_forms = [
        {"fen": fen, "forms": spoken_forms(chess.Board(fen))}
        for fen in (*BOARDS, AFTER_G6, PROMOTION, CASTLED)
    ]

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
        "forms": all_forms,
        "endpointer": endpointer,
        "ratios": ratios,
    }), encoding="utf-8")
    print(f"{len(clips)} clips, {len(corrections)} corrections, {len(ratios)} ratios -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
