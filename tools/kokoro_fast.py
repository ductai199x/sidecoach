"""The phone's faster Kokoro: the decoder's convolutions in 8 bits, its last stage's kept whole.

On the Pixel 10 Pro XL Kokoro's time is the decoder's convolutions (65% of it, ONNX Runtime's
profile). In 8 bits -- static QDQ, per-channel weights, activations calibrated on coach lines --
the graph runs 0.42-0.46 of real time where the shipped full-precision one runs 0.70-0.76
(2026-10-03, with a stream playing beside it). The last stage of the generator (resblocks 3-5,
the highest resolution) stays in full precision: in 8 bits it whined near 11 kHz (+13.5 dB over
fp32 at 10-12 kHz), which Tai heard as a buzz. Kept whole, the bands match fp32 and Tai could not
tell the two apart in a blind A/B. Nothing else of the graph changes: the source's STFT rewritten
as a convolution changed its last digits, and the decoder's phase (Atan of near-zero bins) turned
that into high-pitched noise.

    uv run --no-sync python tools/kokoro_fast.py [--out data/models/kokoro/kokoro-v1.0-q8.onnx]

It prints the check it must pass against fp32 on lines it was not calibrated on -- the excess at
10-12 kHz while speaking and the log-mel distance -- and refuses to keep a graph that fails it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from kokoro_onnx import Kokoro
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

MODELS = Path(__file__).resolve().parent.parent / "data" / "models" / "kokoro"
FP32 = MODELS / "kokoro-v1.0.onnx"
VOICES = MODELS / "voices-v1.0.bin"
VOICE = "bm_george"
SR = 24000
# The convolutions kept in full precision: the generator's last stage, and its output conv.
KEPT = (
    "/generator/resblocks.3/",
    "/generator/resblocks.4/",
    "/generator/resblocks.5/",
    "conv_post",
)
# What the graph must stay within, against fp32 (the A/B'd graph: +0.1 dB and 1.41 dB).
MAX_HIGH_DB = 1.0
MAX_MEL_DB = 1.6

CALIBRATE = [
    "Excellent: the only move that keeps the extra pawn.",
    "Use your king on f6 against the pawn on h4; the quiet approach wins it.",
    "Which square lets you do that while keeping the rook behind the passed pawn?",
    "That was fine. The centre holds, and the bishop is doing real work on the long diagonal.",
    "This was a miss: Rg8 was the quiet mating attack, taking the g-file and the king's squares.",
    "Spot on.",
    "Exactly.",
    "Best.",
    "Yes, you found it.",
    "That's the move.",
    "Only one move keeps the win; the next best gives most of it back.",
    "Their knight on d4 is loose, and your queen can reach it with check.",
    "Beautiful sacrifice: you cleared the way for a mate in two.",
    "Next time, look for the quiet move that takes squares from the king.",
    "Play Qa8 check. It deflects the rook from the back rank, and mate follows.",
    "The opening was the Sicilian, Najdorf variation, and you kept the balance well.",
    "Careful: after the exchange on e5, their bishop on b4 pins your knight to the king.",
    "Your rook belongs on the open file; the c-file is yours for the taking.",
    "That's mate. Well played.",
    "Hmm, that gives up the pawn on f7 for nothing. What does their queen attack now?",
    "Strong! You saw that the defender was overloaded, guarding both the rook and the back rank.",
    "Take a moment. Both kings are open, so every check matters here.",
    "A quiet move, but a good one: the knight heads for the outpost on d5.",
    "Their threat is Qxh2, mate. Which piece can cover h2 in time?",
]
CHECK = [
    "Look at your checks against the king on g6, especially the squares that take away its "
    "escape routes.",
    "Wonderful: this was the only good move, keeping the mating attack alive while their queen "
    "is far away.",
    "Exactly.",
    "Your bishop on c4 eyes f7, the weakest square in their camp, and their king has not castled.",
]


def inputs_for(tts: Kokoro, text: str) -> dict[str, np.ndarray]:
    """The graph's inputs for one line, as kokoro-onnx builds them."""
    tokens = tts.tokenizer.tokenize(tts.tokenizer.phonemize(text, "en-us"))
    style = tts._style_for(tts.get_voice_style(VOICE), len(tokens))
    return {
        tts._tokens_input: np.array([[0, *tokens, 0]], dtype=tts._input_dtypes[tts._tokens_input]),
        "style": np.asarray(style, dtype=tts._input_dtypes["style"]).reshape(1, -1),
        "speed": np.array([tts._speed_value(1.0)], dtype=tts._input_dtypes["speed"]),
    }


def power(x: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    win = np.hanning(n_fft)
    frames = [x[i : i + n_fft] * win for i in range(0, len(x) - n_fft, hop)]
    return np.abs(np.fft.rfft(np.array(frames), axis=1)) ** 2


def log_mel(p: np.ndarray, mels: int = 80) -> np.ndarray:
    hz = np.linspace(0, SR / 2, p.shape[1])
    mel = 2595 * np.log10(1 + hz / 700)
    edges = np.linspace(0, mel[-1], mels + 2)
    bank = np.array(
        [
            np.clip(
                np.minimum(
                    (mel - edges[i]) / (edges[i + 1] - edges[i]),
                    (edges[i + 2] - mel) / (edges[i + 2] - edges[i + 1]),
                ),
                0,
                None,
            )
            for i in range(mels)
        ]
    )
    return 10 * np.log10(p @ bank.T + 1e-10)


def check(path: Path, tts: Kokoro) -> tuple[float, float]:
    """The excess at 10-12 kHz while speaking, and the log-mel distance, against fp32."""
    ref = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    highs, mels = [], []
    for text in CHECK:
        feed = inputs_for(tts, text)
        a, b = ref.run(None, feed)[0].ravel(), got.run(None, feed)[0].ravel()
        n = min(len(a), len(b))
        pa, pb = power(a[:n]), power(b[:n])
        hz = np.linspace(0, SR / 2, pa.shape[1])
        level = 10 * np.log10(pa.sum(axis=1) + 1e-12)
        loud = level > level.max() - 30
        band = (hz >= 10000) & (hz < 12000)
        highs.append(10 * np.log10(pb[loud][:, band].sum() / pa[loud][:, band].sum()))
        ma, mb = log_mel(pa), log_mel(pb)
        speech = ma > ma.max() - 60
        mels.append(np.abs(ma - mb)[speech].mean())
    return float(np.mean(highs)), float(np.mean(mels))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=MODELS / "kokoro-v1.0-q8.onnx")
    args = ap.parse_args()
    tts = Kokoro.from_session(
        ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"]), str(VOICES)
    )
    feeds = [inputs_for(tts, t) for t in CALIBRATE]

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.items = iter(feeds)

        def get_next(self):  # noqa: ANN201
            return next(self.items, None)

    model = onnx.load(str(FP32), load_external_data=False)
    convs = [
        n.name
        for n in model.graph.node
        if n.op_type == "Conv"
        and n.name.startswith("/decoder/")
        and not any(k in n.name for k in KEPT)
    ]
    quantize_static(
        str(FP32),
        str(args.out),
        Reader(),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv"],
        nodes_to_quantize=convs,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"CalibMovingAverage": True},
    )
    high, mel = check(args.out, tts)
    ok = high <= MAX_HIGH_DB and mel <= MAX_MEL_DB
    size = args.out.stat().st_size // 1_000_000
    print(f"{len(convs)} convolutions in 8 bits -> {args.out} ({size} MB)")
    print(
        f"against fp32: 10-12 kHz {high:+.1f} dB (at most {MAX_HIGH_DB}), "
        f"log-mel {mel:.2f} dB (at most {MAX_MEL_DB})"
    )
    if not ok:
        args.out.unlink()
        sys.exit("FAILED the check: the graph was not kept")


if __name__ == "__main__":
    main()
