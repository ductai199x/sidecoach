"""The phone's faster Kokoro: the decoder's convolutions in 8 bits, its last stage's in fp16.

On the Pixel 10 Pro XL Kokoro's time is the decoder's convolutions (65% of it, ONNX Runtime's
profile). In 8 bits -- static QDQ, per-channel weights, activations calibrated on coach lines --
the graph runs 0.36 of real time where the full-precision one runs 0.70 (2026-10-03). The last
stage of the generator (resblocks 3-5, the highest resolution) is not in 8 bits: there it whined
near 11 kHz (+13.5 dB over fp32 at 10-12 kHz), which Tai heard as a buzz; kept whole, Tai could
not tell the graph from fp32 in a blind A/B. Then, the same arithmetic in fewer passes:

- the 65 instance norms built from small ops (mean, sub, square, mean, +eps, sqrt, div) as one
  InstanceNormalization each, and the Snake's Pow(x, 2) as x * x;
- the last stage's convolutions in fp16 (the Pixel's cores do fp16 at twice the rate): 0.31 of
  real time, its speech measured on the Pixel itself as close to fp32 as the A/B'd graph's
  (+0.1 dB at 10-12 kHz, log-mel 1.41). This desktop's CPU has no fp16 arithmetic, so the check
  below does not see the fp16 part: KokoroBench's `-e dump` makes the phone's speech to measure.

Tried and left out, measured on the Pixel: the STFT as a convolution (the decoder's phase, Atan
of near-zero bins, turned its changed last digits into high-pitched noise), every conv of the
generator in fp16 (+3.6 dB at 10-12 kHz), the whole last stage in fp16 (slower: fp16 elementwise
ops are not fast there), the text encoder in 8 bits (the speech's timing changes), 6 or 7
threads, dynamic scheduling, XNNPACK, NNAPI and WebGPU.

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
from onnx import TensorProto, helper, numpy_helper
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


def constant(m: onnx.ModelProto, name: str) -> np.ndarray | None:
    """An initializer's or a Constant node's value, or None."""
    for t in m.graph.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    for n in m.graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            return numpy_helper.to_array(n.attribute[0].t)
    return None


def channel_counts(path: Path, names: list[str], tts: Kokoro) -> dict[str, int]:
    """Each tensor's channel count, read off one run of the graph at `path`."""
    probe = onnx.load(str(path))
    have = {o.name for o in probe.graph.output}
    probe.graph.output.extend(
        helper.make_empty_tensor_value_info(n) for n in names if n not in have
    )
    s = ort.InferenceSession(probe.SerializeToString(), providers=["CPUExecutionProvider"])
    got = s.run(None, inputs_for(tts, "Exactly, that is the move."))
    out = dict(zip([o.name for o in s.get_outputs()], got, strict=True))
    return {n: int(out[n].shape[1]) for n in names}


def fuse_norms(m: onnx.ModelProto, path: Path, tts: Kokoro) -> tuple[int, int]:
    """The instance norms built from small ops as InstanceNormalization, and Pow(x, 2) as x * x.
    Returns how many of each."""
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    found = []  # (div, x, eps)
    for div in g.node:
        if div.op_type != "Div" or len(div.input) != 2:
            continue
        sub, sqrt = prod.get(div.input[0]), prod.get(div.input[1])
        if not sub or not sqrt or sub.op_type != "Sub" or sqrt.op_type != "Sqrt":
            continue
        mean1, add = prod.get(sub.input[1]), prod.get(sqrt.input[0])
        if not mean1 or mean1.op_type != "ReduceMean" or mean1.input[0] != sub.input[0]:
            continue
        if not add or add.op_type != "Add":
            continue
        mean2, eps = prod.get(add.input[0]), constant(m, add.input[1])
        if not mean2 or mean2.op_type != "ReduceMean" or eps is None:
            continue
        sq = prod.get(mean2.input[0])
        squared = sq is not None and (
            (sq.op_type == "Mul" and list(sq.input) == [sub.output[0]] * 2)
            or (sq.op_type == "Pow" and sq.input[0] == sub.output[0])
        )
        axes = [constant(m, n.input[1]) for n in (mean1, mean2)]
        if not squared or any(a is None or list(np.ravel(a)) != [2] for a in axes):
            continue
        found.append((div, sub.input[0], float(np.ravel(eps)[0])))
    width = channel_counts(path, [x for _, x, _ in found], tts)
    for k, (div, x, eps) in enumerate(found):
        scale, bias = f"inorm_{k}_scale", f"inorm_{k}_bias"
        g.initializer.append(numpy_helper.from_array(np.ones(width[x], np.float32), scale))
        g.initializer.append(numpy_helper.from_array(np.zeros(width[x], np.float32), bias))
        inorm = helper.make_node(
            "InstanceNormalization",
            [x, scale, bias],
            [div.output[0]],
            name=div.name + "_as_inorm",
            epsilon=eps,
        )
        g.node.insert(list(g.node).index(div), inorm)
        g.node.remove(div)
    squares = 0
    for n in list(g.node):
        e = constant(m, n.input[1]) if n.op_type == "Pow" else None
        if e is not None and np.ravel(e).size == 1 and float(np.ravel(e)[0]) == 2.0:
            mul = helper.make_node(
                "Mul", [n.input[0], n.input[0]], list(n.output), name=n.name + "_as_mul"
            )
            g.node.insert(list(g.node).index(n), mul)
            g.node.remove(n)
            squares += 1
    # What only the replaced nodes read goes too.
    outputs = {o.name for o in g.output}
    while True:
        read = {i for n in g.node for i in n.input} | outputs
        dead = [n for n in g.node if not any(o in read for o in n.output)]
        if not dead:
            break
        for n in dead:
            g.node.remove(n)
    return len(found), squares


def last_stage_in_fp16(m: onnx.ModelProto) -> int:
    """The last stage's full-precision convolutions in fp16, a cast on each side."""
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    k = 0
    for conv in list(g.node):
        if conv.op_type != "Conv" or not any(s in conv.name for s in KEPT[:3]):
            continue
        if conv.input[1] not in inits:
            continue
        for j in (1, 2):
            if j < len(conv.input) and conv.input[j] in inits:
                t = inits[conv.input[j]]
                half = numpy_helper.to_array(t).astype(np.float16)
                g.initializer.append(numpy_helper.from_array(half, t.name + "_fp16"))
                conv.input[j] = t.name + "_fp16"
        x16, y16 = f"half_{k}_in", f"half_{k}_out"
        to16 = helper.make_node(
            "Cast", [conv.input[0]], [x16], name=f"half_{k}_cast_in", to=TensorProto.FLOAT16
        )
        to32 = helper.make_node(
            "Cast", [y16], [conv.output[0]], name=f"half_{k}_cast_out", to=TensorProto.FLOAT
        )
        conv.input[0], conv.output[0] = x16, y16
        at = list(g.node).index(conv)
        g.node.insert(at, to16)
        g.node.insert(at + 2, to32)
        k += 1
    return k


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
    quantized = args.out.with_suffix(".quantized.onnx")
    quantize_static(
        str(FP32),
        str(quantized),
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
    graph = onnx.load(str(quantized))
    norms, squares = fuse_norms(graph, quantized, tts)
    halves = last_stage_in_fp16(graph)
    onnx.save(graph, str(args.out))
    quantized.unlink()
    high, mel = check(args.out, tts)
    ok = high <= MAX_HIGH_DB and mel <= MAX_MEL_DB
    size = args.out.stat().st_size // 1_000_000
    print(
        f"{len(convs)} convolutions in 8 bits, {halves} in fp16, {norms} instance norms fused, "
        f"{squares} squares as products -> {args.out} ({size} MB)"
    )
    print(
        f"against fp32: 10-12 kHz {high:+.1f} dB (at most {MAX_HIGH_DB}), "
        f"log-mel {mel:.2f} dB (at most {MAX_MEL_DB})"
    )
    if not ok:
        args.out.unlink()
        sys.exit("FAILED the check: the graph was not kept")


if __name__ == "__main__":
    main()
