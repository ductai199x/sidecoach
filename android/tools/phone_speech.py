"""The pieces the phone speaks a line in: the reference `Kokoro.plan` is ported from, and the
plan measured on 2026-09-23 (Tai: pieces on the phone only, the desktop speaks whole lines).

On the desktop a line is one pass on the GPU in a tenth of a second. On the phone's CPU a pass
takes about half as long as the speech, so a whole line starts only once all of it is made. In
pieces, each is spoken while the next is made:

- a line of under 3.5 s of speech (56 ms a phoneme) stays whole, batched as kokoro-onnx batches
  it: the desktop's own audio. In Tai's blind test the short lines sounded better whole;
- a longer line is cut where the text pauses: after a stop, a colon, a semicolon or a comma in
  its phonemes, and at a dash in its text (the phonemizer drops dashes, so each dash-separated
  part is phonemized on its own). Its clauses are then merged into few pieces: the first until
  it has 16 phonemes (about a second), each after that up to twice the one before, and a tail
  of less than half the piece before joins it. A cut never falls between two words of a
  clause. A clause too long for one pass is cut as kokoro-onnx cuts it.

Over the coach's lines at the Pixel's speed this took the first audio from 2.02 s to 1.14 s on
average (p90 3.39 -> 1.59) with no loss of naturalness (UTMOS22 +0.003 on the cut lines) and 9
of 779 words heard differently.

    plan(tokenizer, text, speed) -> [(phonemes, pause after), ...]
"""

from __future__ import annotations

import re

from kokoro_onnx.chunker import pause_after, split_phonemes
from kokoro_onnx.tokenizer import Tokenizer

MS_PER_PHONEME = 56.0
WHOLE_BELOW_S = 3.5
FIRST_AT_LEAST = 16
MAX_PHONEME_LENGTH = 510
DASH = re.compile(r"\s+(?:--|—|–)\s+")
CLAUSE = re.compile(r"(?<=[.!?…,;:])\s+")


def phonemes(tokenizer: Tokenizer, text: str, lang: str) -> str:
    """`Kokoro._prepare`'s phonemes: the tokenizer's, whitespace runs made single spaces."""
    return " ".join(tokenizer.phonemize(text, lang).split())


def batches(phonemes: str, sentence_pause: float, clause_pause: float) -> list[tuple[str, float]]:
    """`Kokoro._prepare`'s batches, each with the pause after it."""
    batched = split_phonemes(phonemes)
    if not batched:
        return []
    return [(b, pause_after(b, sentence_pause, clause_pause)) for b in batched[:-1]] + [(batched[-1], 0.0)]


def plan(
    tokenizer: Tokenizer,
    text: str,
    speed: float = 1.0,
    lang: str = "en-us",
    sentence_pause: float = 0.25,
    clause_pause: float = 0.1,
) -> list[tuple[str, float]]:
    whole = phonemes(tokenizer, text, lang)
    if len(whole) * MS_PER_PHONEME / 1000 / speed < WHOLE_BELOW_S:
        return batches(whole, sentence_pause, clause_pause)
    atoms: list[str] = []
    for part in DASH.split(text):
        if not part.strip():
            continue
        for atom in CLAUSE.split(phonemes(tokenizer, part, lang)):
            if atom.strip():
                atoms.extend(split_phonemes(atom) if len(atom) > MAX_PHONEME_LENGTH else [atom])
    if not atoms:
        return batches(whole, sentence_pause, clause_pause)
    first, i = atoms[0], 1
    while len(first) < FIRST_AT_LEAST and i < len(atoms) and len(first) + 1 + len(atoms[i]) <= MAX_PHONEME_LENGTH:
        first += " " + atoms[i]
        i += 1
    pieces = [first]
    while i < len(atoms):
        piece = atoms[i]
        i += 1
        while i < len(atoms) and len(piece) + 1 + len(atoms[i]) <= min(MAX_PHONEME_LENGTH, 2 * len(pieces[-1])):
            piece += " " + atoms[i]
            i += 1
        pieces.append(piece)
    if (
        len(pieces) > 1
        and len(pieces[-1]) < len(pieces[-2]) // 2
        and len(pieces[-2]) + 1 + len(pieces[-1]) <= MAX_PHONEME_LENGTH
    ):
        last = pieces.pop()
        pieces[-1] = pieces[-1] + " " + last
    return [(p, pause_after(p, sentence_pause, clause_pause)) for p in pieces[:-1]] + [(pieces[-1], 0.0)]
