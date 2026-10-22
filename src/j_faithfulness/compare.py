"""Comparing outcomes (DESIGN.md, Comparing outcomes)."""

from __future__ import annotations

import math


def renormalize(probs: dict[str, float]) -> dict[str, float]:
    """The option probabilities as a distribution over the options."""
    total = sum(probs.values())
    return {o: p / total for o, p in probs.items()}


def sq_dist(p: dict[str, float], q: dict[str, float]) -> float:
    """Squared distance between two (renormalized) answer distributions: 0 if
    they agree, 2 if they put all mass on different options."""
    p, q = renormalize(p), renormalize(q)
    return sum((p[o] - q[o]) ** 2 for o in p)


def brier(probs: dict[str, float], d: str) -> float:
    """Brier score of the (renormalized) answer distribution against the
    original decision ``d``: 0 if all mass is on ``d``, 2 if all is on one
    other option."""
    return sq_dist(probs, {o: float(o == d) for o in probs})


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p ‖ q) between the (renormalized) answer distributions, in nats."""
    p, q = renormalize(p), renormalize(q)
    return sum(p[o] * math.log(p[o] / q[o]) for o in p if p[o] > 0)
