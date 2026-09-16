"""One ONNX runtime for both voice models, on the CPU or the GPU as configured.

Kokoro (speech) and Parakeet (hearing) are both ONNX graphs, so they share one runtime and one
way of choosing a device: `auto` takes the GPU when onnxruntime's CUDA provider actually
initialises, `cuda` insists on it, `cpu` never touches the GPU. onnxruntime-gpu >= 1.27 is
built for CUDA 13 and finds its DLLs in the nvidia-* wheels only after `preload_dlls()`, so
that runs once, before the first session, and never again.

Sessions are created once per model, by the providers, and live for the process: nothing here
loads a model per call, and switching a Kokoro voice never reloads its graph.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEVICES = ("auto", "cpu", "cuda")
_CUDA = "CUDAExecutionProvider"
_CPU = "CPUExecutionProvider"

_lock = threading.Lock()
_preloaded = False
_cuda_ok: bool | None = None  # whether a CUDA session has actually initialised here


@dataclass(frozen=True)
class Device:
    """Where a model ended up, and why, for the log and the status strip."""

    name: str  # "cpu" or "cuda"
    detail: str


def preload() -> None:
    """Load the CUDA / cuDNN DLLs from the nvidia wheels, once. Harmless without a GPU."""
    global _preloaded
    with _lock:
        if _preloaded:
            return
        _preloaded = True
        try:
            import onnxruntime as ort

            preload_dlls = getattr(ort, "preload_dlls", None)
            if preload_dlls is not None:
                preload_dlls()
        except Exception as exc:  # noqa: BLE001 - no CUDA is a normal state, not an error
            log.info("CUDA DLLs not preloaded: %s", exc)


def cuda_available() -> bool:
    """Whether the CUDA provider is present AND a session can be created on it."""
    global _cuda_ok
    with _lock:
        if _cuda_ok is not None:
            return _cuda_ok
    preload()
    try:
        import onnxruntime as ort

        # The provider can be listed yet fail to initialise (missing DLLs); the only
        # proof is a session. A trivial graph keeps the probe cheap.
        ok = _CUDA in ort.get_available_providers() and _probe_cuda(ort)
    except Exception as exc:  # noqa: BLE001
        log.info("CUDA probe failed: %s", exc)
        ok = False
    with _lock:
        _cuda_ok = ok
    return ok


def _probe_cuda(ort: Any) -> bool:
    try:
        from onnx import TensorProto, helper  # type: ignore[reportMissingImports]
    except ImportError:
        # No `onnx` package to build a probe graph: trust the provider list.
        return True
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    session = ort.InferenceSession(model.SerializeToString(), providers=[_CUDA, _CPU])
    return session.get_providers()[0] == _CUDA


def providers_for(device: str) -> tuple[list[str], Device]:
    """The onnxruntime provider list for a requested device, and where that lands.

    `cuda` (and its synonym `auto`) takes the GPU when the CUDA provider initialises and
    falls back to the CPU otherwise, saying so; `cpu` is CPU only. Nothing here fails for
    want of a GPU.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if device == "cpu":
        return [_CPU], Device("cpu", "cpu (as configured)")
    if cuda_available():
        return [_CUDA, _CPU], Device("cuda", "cuda")
    return [_CPU], Device("cpu", "cpu (CUDA not available, fell back)")


def session(path: Path | str, device: str, **options: Any) -> tuple[Any, Device]:
    """An InferenceSession for `path` on `device`. Create it once and keep it."""
    import onnxruntime as ort

    providers, where = providers_for(device)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # errors only; the memcpy notices on CUDA are not actionable
    for key, value in options.items():
        setattr(opts, key, value)
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    active = sess.get_providers()[0]
    if active != providers[0]:
        where = Device("cpu", f"cpu (session fell back from {providers[0]})")
    return sess, where
