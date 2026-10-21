"""Comparing outcomes (DESIGN.md, Comparing outcomes)."""

from __future__ import annotations

import math


def renormalize(probs: dict[str, float]) -> dict[str, float]:
    """The option probabilities as a distribution over the options."""
    total = sum(probs.values())
    return {o: p / total for o, p in probs.items()}


def brier(probs: dict[str, float], d: str) -> float:
    """Brier score of the (renormalized) answer distribution against the
    original decision ``d``: 0 if all mass is on ``d``, 2 if all is on one
    other option."""
    p = renormalize(probs)
    return sum((q - (o == d)) ** 2 for o, q in p.items())


def log_prob(probs: dict[str, float], d: str) -> float:
    """log P(d) under the renormalized distribution."""
    return math.log(renormalize(probs)[d])
