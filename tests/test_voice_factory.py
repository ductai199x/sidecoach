"""The voice factory always returns something with a `say`, and voice never raises."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

from sidecoach.config import Settings, VoiceSettings
from sidecoach.voice import make_recogniser, make_tts
from sidecoach.voice.base import NullTTS
from sidecoach.voice.kokoro import KokoroTTS
from sidecoach.voice.stt import PARAKEET_MODEL, NullSTT, ParakeetSTT


def test_voice_turned_off_is_silent_and_deaf_with_a_reason() -> None:
    tts = make_tts(Settings(), off=True)
    assert isinstance(tts, NullTTS)
    ready, reason = tts.healthcheck()
    assert ready is False and reason
    # A silent provider must accept calls and do nothing, never raise.
    tts.say("Nf3 is a good move")
    tts.stop()
    stt = make_recogniser(Settings(), off=True)
    assert isinstance(stt, NullSTT)
    ready, reason = stt.healthcheck()
    assert ready is False and "off" in reason
    assert stt.transcribe(np.zeros(1600, dtype=np.float32)) == ""


def test_kokoro_is_built_with_the_configured_voice_and_model_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(voice="am_adam", speed=1.1))
    tts = make_tts(settings)
    assert isinstance(tts, KokoroTTS)
    assert tts.voice == "am_adam"
    assert tts.speed == 1.1
    assert tts.model_dir == tmp_path / "models" / "kokoro"
    # No model files and (in CI) no extras: healthcheck reports not-ready without loading, and
    # neither healthcheck nor stop raises.
    ready, reason = tts.healthcheck()
    assert ready is False and reason
    tts.stop()


def test_speech_recognition_is_parakeet_on_the_configured_device() -> None:
    stt = make_recogniser(Settings(voice=VoiceSettings(stt_device="cpu")))
    assert isinstance(stt, ParakeetSTT) and stt.device_wanted == "cpu"
    # It states its device before loading anything.
    ready, detail = stt.healthcheck()
    if find_spec("onnx_asr") is None:
        # Without the voice group (CI installs dev and panel only) the ear says so.
        assert not ready and "onnx-asr" in detail
    else:
        assert ready and detail == f"{PARAKEET_MODEL}, cpu (not loaded yet)"


def test_explicit_model_dir_overrides_the_default() -> None:
    settings = Settings(voice=VoiceSettings(model_dir="C:/models/kokoro"))
    assert settings.voice_model_dir() == Path("C:/models/kokoro")


def test_a_failed_kokoro_say_is_a_noop_and_starts_no_worker(tmp_path: Path) -> None:
    tts = KokoroTTS(tmp_path)
    tts._failed = True  # simulate a backend that could not load
    tts.say("Qh5+")
    assert tts._worker is None


def test_copies_of_espeak_left_by_dead_processes_are_swept_and_nothing_else(tmp_path: Path) -> None:
    """The phonemizer's copies of espeak's library, left in Temp by processes that never reached
    a clean exit (two hundred in Tai's, 2026-10-04): a folder holding that one file goes; a
    copy still loaded (an open file cannot be deleted on Windows), a folder holding anything
    else, and a folder not named as Python names them stay."""
    from sidecoach.voice.kokoro import ESPEAK_COPY, sweep_espeak_copies

    def folder(name: str, *files: str) -> Path:
        (tmp_path / name).mkdir()
        for f in files:
            (tmp_path / name / f).write_bytes(b"x")
        return tmp_path / name

    stale = [folder("tmpab12cd34", ESPEAK_COPY), folder("tmp_9xyz", ESPEAK_COPY)]
    busy = folder("tmpbusy0001", ESPEAK_COPY)
    kept = [folder("tmpmore0002", ESPEAK_COPY, "profile.sqlite"), folder("mine", ESPEAK_COPY)]
    with (busy / ESPEAK_COPY).open("rb"):
        assert sweep_espeak_copies(tmp_path) == 2
    assert not any(p.exists() for p in stale)
    assert busy.exists() and all((p / ESPEAK_COPY).exists() for p in kept)


def test_linux_copies_are_swept_but_never_one_a_live_process_has_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Linux the copy is libespeak-ng.so (eight piled up in /tmp after a few launches, the
    Linux audit 2026-10-13), and Linux, unlike Windows, deletes a loaded file: a copy some
    process has mapped (/proc/*/maps) is left to it."""
    from sidecoach.voice import kokoro

    for name in ("tmpdead0001", "tmplive0002"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "libespeak-ng.so").write_bytes(b"x")
    live = tmp_path / "tmplive0002" / "libespeak-ng.so"
    monkeypatch.setattr(kokoro.sys, "platform", "linux")
    monkeypatch.setattr(kokoro, "_mapped_libraries", lambda: {str(live)})
    assert kokoro.sweep_espeak_copies(tmp_path) == 1
    assert not (tmp_path / "tmpdead0001").exists() and live.exists()


def test_the_mapped_files_are_read_past_anonymous_memory(tmp_path: Path) -> None:
    """/proc/<pid>/maps as Linux writes it: anonymous memory has no path, only the padding after
    its inode, and the reader failed on that line -- every Linux start, so the voice went
    silent ("list index out of range", Ubuntu 22.04, 2026-10-14). A path may hold spaces."""
    from sidecoach.voice import kokoro

    (tmp_path / "1234").mkdir()
    (tmp_path / "1234" / "maps").write_text(
        "5581c3a00000-5581c3a28000 r--p 00000000 08:01 1835106                    /usr/bin/cat\n"
        "7f2b1c000000-7f2b1c021000 rw-p 00000000 00:00 0                          \n"
        "7f2b1c400000-7f2b1c421000 r-xp 00001000 08:01 42   /tmp/tmpab12/libespeak-ng.so\n"
        "7f2b1c800000-7f2b1c821000 r--p 00000000 08:01 43   /home/a/My Apps/lib x.so\n"
        "7ffd5a3c1000-7ffd5a3e2000 rw-p 00000000 00:00 0                          [stack]\n",
        encoding="utf-8",
    )
    (tmp_path / "self").mkdir()  # not a process number: not read
    (tmp_path / "self" / "maps").write_text("garbage\n", encoding="utf-8")
    assert kokoro._mapped_libraries(tmp_path) == {
        "/usr/bin/cat",
        "/tmp/tmpab12/libespeak-ng.so",
        "/home/a/My Apps/lib x.so",
        "[stack]",
    }
