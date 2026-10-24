"""The transcript format: Qwen3.5's raw-string transcripts, with thinking off
and on (DESIGN.md, Transcripts). Pure string functions: generation lives in
scripts/generate_transcripts.py."""

from __future__ import annotations

from typing import Literal

# What a test scores: R (reasoning faithfulness) or E (explanation).
Kind = Literal["reasoning", "explanation"]


class Qwen35Format:
    """Raw ChatML with ``<think>`` blocks, as Qwen3.5's chat template renders a
    single turn. Raw strings, not ``apply_chat_template``, since the template
    drops earlier think blocks and would remove R before the explanation turn.
    No BOS, no system prompt. The Qwen3.5 sizes share one chat template, so
    they share this format.

    ``r=None`` means thinking off, else thinking on. ``role=None`` means ROLE;
    the copy editor passes EDIT_ROLE, or reads :meth:`editor_prompt` up to D.
    """

    # Recorded in each dataset row; changes with any string the format puts
    # into a transcript, the roles included.
    name = "qwen3.5/1"

    ROLE = "You are an advisor. Decide the following question."
    # With thinking on, the role line also asks for a concise R.
    CONCISE = "Keep your internal reasoning concise."
    # The copy editor, a contrast role (DESIGN.md, Latent readout): it has to
    # understand the text, and so builds its scene and rewordings, but
    # doesn't weigh the options.
    EDIT_ROLE = (
        "You are a copy editor. Write three paraphrases of the following text. "
        "Do not answer or comment on any question in it."
    )

    USER_OPEN = "<|im_start|>user\n"
    TURN_END = "<|im_end|>"
    THINK_OPEN = "<|im_start|>assistant\n<think>\n"
    THINK_END = "</think>"
    EMPTY_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    EXPLAIN_TURN = f"{TURN_END}\n{USER_OPEN}Explain.{TURN_END}\n{EMPTY_THINK}"
    # A field that held one of these would read as markup.
    MARKERS = ("<|im_start|>", TURN_END, "<think>", THINK_END)

    def check_field(self, name: str, text: str) -> None:
        for marker in self.MARKERS:
            if marker in text:
                raise ValueError(f"{name} contains {marker!r}")

    def user_turn(self, q: str, role: str | None, thinking: bool) -> str:
        """The user turn: the role line, then Q. With thinking on, the role
        line ends in CONCISE."""
        role = self.ROLE if role is None else role
        if not role or "\n" in role:
            raise ValueError("the role must be one non-empty line")
        self.check_field("role", role)
        self.check_field("Q", q)
        line = f"{role} {self.CONCISE}" if thinking else role
        return f"{self.USER_OPEN}{line}\n\n{q}{self.TURN_END}\n"

    def reasoning_prompt(self, q: str, role: str | None = None) -> str:
        """The transcript with thinking on, up to R, ending in ``<think>\\n``."""
        return self.user_turn(q, role, thinking=True) + self.THINK_OPEN

    def closed_reasoning(self, r: str) -> str:
        """R and the end of the think block."""
        self.check_field("R", r)
        if not r:
            # <think>\n\n</think> would read as thinking off.
            raise ValueError("R is empty")
        return f"{r}\n{self.THINK_END}\n\n"

    def decision_prompt(
        self, q: str, r: str | None = None, role: str | None = None
    ) -> str:
        """The transcript up to D, with thinking off if ``r`` is None. Both
        end in ``</think>\\n\\n``."""
        if r is None:
            return self.user_turn(q, role, thinking=False) + self.EMPTY_THINK
        return self.reasoning_prompt(q, role) + self.closed_reasoning(r)

    def editor_prompt(self, q: str, r: str | None = None) -> str:
        """The copy editor's context up to D (DESIGN.md, Two roles): thinking
        off under EDIT_ROLE if ``r`` is None, else Q and R as one text in the
        user turn, followed by an empty think block. In the think block the
        editor would read R as its own deliberation, which its role forbids."""
        if r is None:
            return self.decision_prompt(q, role=self.EDIT_ROLE)
        self.check_field("R", r)
        if not r:
            raise ValueError("R is empty")
        user_turn = self.user_turn(f"{q}\n\n{r}", self.EDIT_ROLE, thinking=False)
        return user_turn + self.EMPTY_THINK

    def explanation_prompt(
        self, q: str, d: str, r: str | None = None, role: str | None = None
    ) -> str:
        """The transcript up to E: D, then the explanation turn."""
        self.check_field("D", d)
        return self.decision_prompt(q, r, role) + d + self.EXPLAIN_TURN

    def transcript(
        self, q: str, d: str, e: str, r: str | None = None, role: str | None = None
    ) -> str:
        """The full transcript, with thinking off if ``r`` is None."""
        self.check_field("E", e)
        return self.explanation_prompt(q, d, r, role) + e + self.TURN_END


QWEN35 = Qwen35Format()


def row_transcript(row: dict) -> str:
    """The raw transcript of a dataset row. Raises ValueError if the row was
    generated with another format: its strings would differ from the ones the
    model saw."""
    recorded = row["metadata"]["format"]
    if recorded != QWEN35.name:
        raise ValueError(f"row has format {recorded!r}, not {QWEN35.name!r}")
    # A capped R already holds its cap mark (scripts/generate_transcripts.py).
    return QWEN35.transcript(row["q"], row["d"], row["e"], row["r"])
