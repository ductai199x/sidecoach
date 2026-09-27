"""The earlier dialogue, folded into a summary so the harness stays flat in a long game.

Every `every` plies the coach folds what was said earlier than the last `keep` plies into
one block at the top of the record. The summariser sees only the dialogue -- the previous
summary and the lines being folded, each with the move it followed -- never the position,
so it can add no fact; the record and the position stay lossless. The last `keep` plies
stay verbatim, so the coach still knows its own last words.
"""

from __future__ import annotations

import threading
from contextlib import nullcontext

from sidecoach.coach.conversation import Conversation
from sidecoach.llm.base import ChatMessage, LLMProvider

SUMMARY_SYSTEM = (
    "You keep a running summary of what a chess coach and a player have said to each other "
    "during a game. You are given the summary so far and the newest exchanges, each marked "
    "with the move it followed. Write the new summary: what the coach has taught or pointed "
    "at (the ideas, the patterns, the moves it praised or faulted, with move numbers), what "
    "the player asked, and what advice was given -- in the third person ('the coach', 'the "
    "player'), at most 120 words, no new facts about the position, nothing that was not "
    "said. Plain text, no headings."
)


def summary_request(pending: list, summary_before: str | None) -> list[ChatMessage]:
    """The summariser's message for the exchanges being folded (`Conversation.foldable`: the
    move each followed, and the entry), after the summary so far. Pure, so the phone's
    fixture is made from this text rather than from a copy of it (the parity audit)."""
    lines = []
    for after, entry in pending:
        who = "PLAYER" if entry.kind == "ask" else "COACH"
        lines.append(f"after {after}, {who}: {entry.text}")
    content = (
        "Summary so far:\n"
        + (summary_before or "(nothing yet)")
        + "\n\nNewest exchanges:\n"
        + "\n".join(lines)
        + "\n\nThe new summary:"
    )
    return [{"role": "user", "content": content}]


def summarise(
    convo: Conversation,
    provider: LLMProvider,
    *,
    keep: int = 20,
    lock: threading.Lock | threading.RLock | None = None,
) -> bool:
    """Fold the dialogue older than the last `keep` plies into the summary. Returns
    whether anything was folded; a failed call folds nothing. `lock` guards the
    conversation while it is read and folded; the model call runs outside it."""
    guard = lock if lock is not None else nullcontext()
    with guard:
        pending = convo.foldable(keep)
        summary_before = convo.summary
    if not pending:
        return False
    messages = summary_request(pending, summary_before)
    try:
        # A rewrite, not a reasoning task: no thinking, where the provider can turn it off.
        completion = provider.chat(
            SUMMARY_SYSTEM, messages, max_tokens=400, temperature=0.1, thinking=False
        )
    except Exception:  # noqa: BLE001
        return False
    text = (completion.text or "").strip()
    if not text:
        return False
    with guard:
        convo.fold([e for _, e in pending], text)
    return True
