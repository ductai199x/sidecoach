"""The pickers the settings window offers (Tai, 2026-09-21): the English voices under
readable labels, with a preview line; the coach's manner as a choice; the AI Coach page
in sections by what a field applies to."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sidecoach.coach.prompt import DEFAULT_PERSONA, PERSONAS, persona_text, system_message
from sidecoach.config import CoachSettings, LLMSettings, Settings, VoiceSettings
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app
from sidecoach.ui.settings_api import describe
from sidecoach.voice.voices import ENGLISH_VOICES, english_voices, is_english, voice_label


def _silent() -> VoiceSettings:
    return VoiceSettings()


# The persona the coach was evaluated with, as it read before the choice existed
# (docs/eval-results.md): the default must still be this, word for word.
EVALUATED = (
    "You are a chess coach sitting beside a player, watching their game as it is played. "
    "You are warm but direct, like a good club coach: encouraging, never gushing, and "
    "honest about mistakes. Your aim is not to hand over answers but to build the player's "
    "own eyesight for tactics and ideas, so they see the pattern themselves next time. "
    "This is one continuous game, not a string of puzzles: connect this moment to earlier "
    "ones when it helps the player see a recurring idea, and build on what you have "
    "already said."
)


def test_voice_labels_read_as_accent_sex_and_name() -> None:
    assert voice_label("af_sarah") == "American Female, Sarah"
    assert voice_label("bm_george") == "British Male, George"
    assert voice_label("am_michael") == "American Male, Michael"
    assert voice_label("bf_emma") == "British Female, Emma"
    # Outside the scheme, the name is shown as it is.
    assert voice_label("jf_alpha") == "jf_alpha" and not is_english("jf_alpha")
    assert all(is_english(v) for v in ENGLISH_VOICES)
    # American and British, female and male, all four kinds offered.
    assert {v[:2] for v in ENGLISH_VOICES} == {"af", "am", "bf", "bm"}


def test_the_voice_list_falls_back_to_the_known_names(tmp_path: Path) -> None:
    assert english_voices(None) == list(ENGLISH_VOICES)
    assert english_voices(tmp_path / "missing.bin") == list(ENGLISH_VOICES)
    assert english_voices(tmp_path / "not-a-file.bin") == list(ENGLISH_VOICES)


def test_the_default_persona_is_the_evaluated_one_and_the_choices_match() -> None:
    assert persona_text(DEFAULT_PERSONA) == EVALUATED
    assert persona_text(None) == EVALUATED and persona_text("no-such") == EVALUATED
    assert system_message(with_corrections=False).startswith(EVALUATED)
    playful = system_message(with_corrections=False, persona="playful")
    rest = system_message(with_corrections=False)[len(EVALUATED) :]
    assert "playful" in playful and playful.endswith(rest)
    # The settings' choices are the personas, in the same order.
    extra = CoachSettings.model_fields["persona"].json_schema_extra
    assert isinstance(extra, dict) and list(extra["choices"]) == list(PERSONAS)  # type: ignore[arg-type]


def test_the_settings_describe_the_pickers_and_the_ai_coach_sections(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=_silent())
    described = describe(settings)
    assert described["voices"] == list(ENGLISH_VOICES)
    assert described["personas"] == {name: label for name, (label, _) in PERSONAS.items()}
    props = described["schema"]["$defs"]["VoiceSettings"]["properties"]
    assert props["voice"]["kind"] == "voice"
    coach = described["schema"]["$defs"]["CoachSettings"]["properties"]
    assert coach["persona"]["kind"] == "persona"
    llm = described["schema"]["$defs"]["LLMSettings"]["properties"]
    groups = {name: prop.get("group") for name, prop in llm.items()}
    assert groups["provider"] == "provider" and groups["base_url"] == "local"
    assert groups["api_key_env"] == "service" and groups["price_prompt_per_m"] == "service"
    assert groups["model"] == "general" and groups["temperature"] == "general"
    assert set(groups.values()) == {"provider", "local", "service", "general"}
    assert set(LLMSettings.model_fields) == set(groups), "every field is in a section"


def test_the_preview_line_says_whether_the_voice_could_speak(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=_silent())
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    client = TestClient(build_app(engine))
    # The silent provider cannot speak: the button is told so, nothing breaks.
    assert client.post("/api/say", json={"voice": "bf_emma"}).json() == {"ok": False}
    assert engine.voice == settings.voice.voice, "the coach's own voice is untouched"
