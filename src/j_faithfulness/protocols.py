"""Raw-string transcripts (see DESIGN.md, Conversation protocols)."""

from __future__ import annotations

EMPTY_THINK = "<think>\n\n</think>\n\n"


def user_turn(text: str) -> str:
    return f"<|im_start|>user\n{text}<|im_end|>\n"


def decision_prompt(q: str) -> str:
    """Protocol 1 up to D: Q, then the assistant turn with thinking off. Its last
    token is where D is produced."""
    return user_turn(q) + "<|im_start|>assistant\n" + EMPTY_THINK
