"""Comparing outcomes (DESIGN.md, Comparing outcomes)."""

from __future__ import annotations

import math


def renormalize(probs: dict[str, float]) -> dict[str, float]:
    """The option probabilities as a distribution over the options."""
    total = sum(probs.values())
    return {o: p / total for o, p in probs.items()}


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p ‖ q) between two distributions over the same options, in nats."""
    return sum(p[o] * math.log(p[o] / q[o]) for o in p if p[o] > 0)


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen–Shannon divergence between the (renormalized) answer
    distributions, in nats: symmetric, 0 if they agree, ln 2 if they put all
    mass on different options."""
    p, q = renormalize(p), renormalize(q)
    m = {o: (p[o] + q[o]) / 2 for o in p}
    return (kl(p, m) + kl(q, m)) / 2
