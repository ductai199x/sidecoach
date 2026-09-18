"""
Utilities to prepare ChessQA-Benchmark prompts for evaluation.

The dataset questions contain placeholder tokens:
  - ``CONTEXT_PLACEHOLDER``: optionally replace with descriptive context.
  - ``FORMAT_EXAMPLE_PLACEHOLDER``: optionally replace with a sample answer format.

The helpers below make it easy to fill those placeholders consistently before
calling an LLM.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import chess


@dataclass
class PromptConfig:
    """Configuration for prompt formatting."""

    add_context: bool = True
    """Inject piece arrangement and legal moves into the prompt when true."""

    format_example_index: int = 0
    """Pick which entry from ``format_examples`` to include (0-based)."""


def _clean_fen(fen: str) -> str:
    """Strip move annotations that follow the FEN with a pipe separator."""
    return fen.split("|", 1)[0].strip()


def get_context(fen: str) -> str:
    """
    Build a textual description for the provided FEN.

    Returns piece arrangement and all legal moves. When the FEN is invalid the
    function returns an empty string instead of raising.
    """
    fen_clean = _clean_fen(fen)
    try:
        board = chess.Board(fen_clean)
    except Exception:
        try:
            board = chess.Board(" ".join(fen_clean.split()))
        except Exception:
            return ""

    piece_names = {
        chess.PAWN: "Pawn",
        chess.KNIGHT: "Knight",
        chess.BISHOP: "Bishop",
        chess.ROOK: "Rook",
        chess.QUEEN: "Queen",
        chess.KING: "King",
    }

    pieces: dict[str, list[str]] = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        color = "White" if piece.color == chess.WHITE else "Black"
        key = f"{color} {piece_names[piece.piece_type]}"
        pieces.setdefault(key, []).append(chess.square_name(square))

    arrangement_parts = [f"{k}: {sorted(v)}" for k, v in pieces.items()]
    arrangement = ", ".join(arrangement_parts)
    legal_moves = ", ".join(sorted(move.uci() for move in board.legal_moves))
    return f"Piece arrangement: {arrangement}\nLegal moves: {legal_moves}\n\n"


def _select_format_example(format_examples: Sequence[str] | None, index: int) -> str:
    if not format_examples:
        return ""
    if index < 0 or index >= len(format_examples):
        index = 0
    return format_examples[index]


def _ensure_question_fields(row: Mapping[str, Any]) -> tuple[str, Sequence[str]]:
    question = row.get("question", "")
    if not isinstance(question, str):
        raise TypeError("Expected 'question' to be a string")

    raw_examples = row.get("format_examples", [])
    if isinstance(raw_examples, str):
        try:
            raw_examples = json.loads(raw_examples)
        except json.JSONDecodeError:
            raw_examples = [raw_examples]

    if not isinstance(raw_examples, Sequence):
        raise TypeError("Expected 'format_examples' to be a sequence")
    return question, raw_examples


def format_prompt(
    row: Mapping[str, Any],
    config: PromptConfig | None = None,
) -> str:
    """
    Replace placeholders in the dataset question.

    Parameters
    ----------
    row:
        A single dataset row (e.g., from the Hugging Face ``datasets`` library).
    config:
        Prompt configuration. When omitted defaults to ``PromptConfig()``.
    """
    config = config or PromptConfig()
    question, format_examples = _ensure_question_fields(row)

    formatted = question
    if config.add_context:
        fen = row.get("input", "")
        context = get_context(fen) if isinstance(fen, str) else ""
        formatted = formatted.replace("CONTEXT_PLACEHOLDER", context)
    else:
        formatted = formatted.replace("CONTEXT_PLACEHOLDER", "")

    example_text = _select_format_example(format_examples, config.format_example_index)
    formatted = formatted.replace("FORMAT_EXAMPLE_PLACEHOLDER", example_text)
    return formatted


__all__ = ["PromptConfig", "format_prompt", "get_context"]
