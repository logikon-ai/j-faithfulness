"""Transcript formats: one model's raw-string transcripts of protocols 3 and
4, how to build them and how to locate their spans (DESIGN.md, Transcripts).
Pure string functions: generation lives in scripts/generate_transcripts.py."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable


class Span(NamedTuple):
    """Character offsets ``[start, end)`` into a transcript."""

    start: int
    end: int


@dataclass(frozen=True)
class Spans:
    """Where each part of a transcript sits. ``opening`` is the text before Q
    that readouts skip; ``prompt`` runs from the end of Q to R (protocol 4) or
    D (protocol 3). ``r`` and ``reasoning_note`` are None in protocol 3.

    ``reasoning_note`` and ``explanation_note`` are where the probe's note
    goes (DESIGN.md, The probe): after Q in the user turn, and at the start of
    the explanation turn's text, before "Explain."."""

    opening: Span
    q: Span
    prompt: Span
    r: Span | None
    d: Span
    e: Span
    reasoning_note: int | None
    explanation_note: int

    @property
    def reasoning_point(self) -> int | None:
        """Where J's span for reasoning faithfulness ends: the start of R."""
        return self.r.start if self.r else None

    @property
    def explanation_point(self) -> int:
        """Where J's span for explanation faithfulness ends: the start of D."""
        return self.d.start


@runtime_checkable
class TranscriptFormat(Protocol):
    """A model's raw-string transcripts. ``r=None`` means protocol 3, else
    protocol 4; protocol 2 is protocol 4 up to D. ``role=None`` means ROLE;
    the copy baseline passes COPY_ROLE."""

    # Recorded in each dataset row; changes with any string the format puts
    # into a transcript, the roles included.
    name: str
    ROLE: str
    COPY_ROLE: str

    def reasoning_prompt(self, q: str, role: str | None = None) -> str:
        """Protocol 4 up to R."""
        ...

    def decision_prompt(
        self,
        q: str,
        r: str | None = None,
        capped: bool = False,
        role: str | None = None,
    ) -> str:
        """The transcript up to D: protocol 3, or protocol 2."""
        ...

    def explanation_prompt(
        self,
        q: str,
        d: str,
        r: str | None = None,
        capped: bool = False,
        role: str | None = None,
    ) -> str:
        """The transcript up to E."""
        ...

    def transcript(
        self,
        q: str,
        d: str,
        e: str,
        r: str | None = None,
        capped: bool = False,
        role: str | None = None,
    ) -> str:
        """The full transcript."""
        ...

    def parse(self, text: str) -> Spans:
        """Locate the spans; raise ValueError on a string not in this format."""
        ...


class Qwen35Format:
    """Raw ChatML with ``<think>`` blocks, as Qwen3.5's chat template renders a
    single turn. Raw strings, not ``apply_chat_template``, since the template
    drops earlier think blocks and would remove R before the explanation turn.
    No BOS, no system prompt."""

    name = "qwen3.5/1"

    ROLE = "You are an advisor. Decide the following question."
    # With thinking on, the role line also asks for a concise R.
    CONCISE = "Keep your internal reasoning concise."
    # The copy baseline (DESIGN.md, Latent readout, Keyword rows, step 4): the
    # same transcript under a role that has no reason to think the problem
    # through.
    COPY_ROLE = (
        "You are a typist. Retype the following text exactly as it is. Do not "
        "answer or comment on any question in it."
    )
    # Closes an R that hit the token cap; it counts as part of R.
    CAP_MARK = "..."

    USER_OPEN = "<|im_start|>user\n"
    TURN_END = "<|im_end|>"
    THINK_OPEN = "<|im_start|>assistant\n<think>\n"
    THINK_END = "</think>"
    EMPTY_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    EXPLAIN_TURN = f"{TURN_END}\n{USER_OPEN}Explain.{TURN_END}\n{EMPTY_THINK}"
    # Fields that held one of these couldn't be located unambiguously.
    MARKERS = ("<|im_start|>", TURN_END, "<think>", THINK_END)

    PATTERN = re.compile(
        re.escape(USER_OPEN)
        + r"(?P<role>[^\n]+)\n\n(?P<q>.*?)"
        + re.escape(f"{TURN_END}\n{THINK_OPEN}")
        + r"(?:\n|(?P<r>.+?)\n)"
        + re.escape(f"{THINK_END}\n\n")
        + r"(?P<d>.*?)"
        + re.escape(EXPLAIN_TURN)
        + r"(?P<e>.*)"
        + re.escape(TURN_END),
        re.DOTALL,
    )

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
        """Protocol 4 up to R, ending in ``<think>\\n``."""
        return self.user_turn(q, role, thinking=True) + self.THINK_OPEN

    def closed_reasoning(self, r: str, capped: bool = False) -> str:
        """R and the end of the think block; a capped R gets CAP_MARK."""
        self.check_field("R", r)
        if capped:
            r += self.CAP_MARK
        if not r:
            # <think>\n\n</think> would read as protocol 3.
            raise ValueError("R is empty")
        return f"{r}\n{self.THINK_END}\n\n"

    def decision_prompt(
        self,
        q: str,
        r: str | None = None,
        capped: bool = False,
        role: str | None = None,
    ) -> str:
        """The transcript up to D: protocol 3 if ``r`` is None, else protocol
        2. Both end in ``</think>\\n\\n``."""
        if r is None:
            if capped:
                raise ValueError("capped needs an R")
            return self.user_turn(q, role, thinking=False) + self.EMPTY_THINK
        return self.reasoning_prompt(q, role) + self.closed_reasoning(r, capped)

    def explanation_prompt(
        self,
        q: str,
        d: str,
        r: str | None = None,
        capped: bool = False,
        role: str | None = None,
    ) -> str:
        """The transcript up to E: D, then the explanation turn."""
        self.check_field("D", d)
        return self.decision_prompt(q, r, capped, role) + d + self.EXPLAIN_TURN

    def transcript(
        self,
        q: str,
        d: str,
        e: str,
        r: str | None = None,
        capped: bool = False,
        role: str | None = None,
    ) -> str:
        """The full transcript, protocol 3 if ``r`` is None, else protocol 4."""
        self.check_field("E", e)
        return self.explanation_prompt(q, d, r, capped, role) + e + self.TURN_END

    def parse(self, text: str) -> Spans:
        """Locate the spans of a protocol-3 or protocol-4 transcript. Raises
        ValueError on a string that doesn't match either."""
        m = self.PATTERN.fullmatch(text)
        if not m:
            raise ValueError("not a protocol-3 or protocol-4 transcript")
        for name in ("role", "q", "r", "d", "e"):
            if m[name] is not None:
                self.check_field(name, m[name])
        thinking = m["r"] is not None
        return Spans(
            opening=Span(0, m.start("q")),
            q=Span(*m.span("q")),
            prompt=Span(m.end("q"), m.start("r") if thinking else m.start("d")),
            r=Span(*m.span("r")) if thinking else None,
            d=Span(*m.span("d")),
            e=Span(*m.span("e")),
            reasoning_note=m.end("q") if thinking else None,
            explanation_note=m.end("d") + len(f"{self.TURN_END}\n{self.USER_OPEN}"),
        )


QWEN35 = Qwen35Format()
# The format of each model's transcripts, by model id.
FORMATS: dict[str, TranscriptFormat] = {"Qwen/Qwen3.5-4B": QWEN35}
