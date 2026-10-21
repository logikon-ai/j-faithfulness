"""LatentReadout → J (one row of ranked words per position) → fenced text."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from j_faithfulness.lens import LatentReadout


@dataclass
class Row:
    """The words of J at one context position, best first."""

    position: int
    words: list[str]


J = list[Row]


def is_word_like(token: str) -> bool:
    """jlens's ``mask_display`` rule: after stripping whitespace, only
    alphanumerics, with ' - ’ allowed inside; no special tokens."""
    s = token.strip()
    if not s or "<|" in s or (s.startswith("<") and s.endswith(">")):
        return False
    return all(
        ch.isalnum() or (0 < i < len(s) - 1 and ch in "'-’") for i, ch in enumerate(s)
    )


def normalize(token: str) -> str:
    """Stripped and NFKC-normalized, so ``" A"``, ``"A"`` and the fullwidth
    ``"Ａ"`` are one word."""
    return unicodedata.normalize("NFKC", token.strip())


def words(text: str) -> set[str]:
    """Casefolded words of ``text``, e.g. of the context for ``drop``."""
    return {w.casefold() for w in re.findall(r"\w+(?:['’-]\w+)*", text)}


def scores(top: list[list[str]]) -> dict[str, float]:
    """Word-like words at one position, scored by summed reciprocal rank over
    the layers (the best-ranked variant counts in each layer)."""
    total: Counter[str] = Counter()
    for ranked in top:
        best: dict[str, int] = {}
        for rank, token in enumerate(ranked):
            word = normalize(token)
            if is_word_like(word):
                best.setdefault(word, rank)
        for word, rank in best.items():
            total[word] += 1 / (rank + 1)
    return dict(total)


def verbalize(
    readout: LatentReadout,
    k: int = 10,
    min_score: float = 0.1,
    max_share: float = 0.5,
    drop: Iterable[str] = (),
    drop_context: bool = True,
) -> J:
    """J: per position, up to ``k`` words scoring at least ``min_score``, best
    first; positions left without words are omitted.

    Left out are words of the context (``drop_context``; at a word of Q the
    lens partly predicts Q's next word), words in ``drop`` (compared
    casefolded), and generic words that pass ``min_score`` at more than one
    position and more than ``max_share`` of them.
    """
    dropped = {w.casefold() for w in drop}
    if drop_context:
        dropped |= words(readout.context)
    per_position = [
        {
            w: s
            for w, s in scores(top).items()
            if s >= min_score and w.casefold() not in dropped
        }
        for top in readout.top
    ]
    n_with = Counter(w for ws in per_position for w in ws)
    generic = {
        w for w, n in n_with.items() if n > 1 and n > max_share * len(per_position)
    }
    j = []
    for position, ws in zip(readout.positions, per_position, strict=True):
        ranked = sorted((w for w in ws if w not in generic), key=lambda w: -ws[w])
        if ranked:
            j.append(Row(position, ranked[:k]))
    return j


def fence(j: J) -> str:
    """J as a ``<latentthought>`` block, one comma-separated row per line."""
    lines = [", ".join(row.words) + "." for row in j]
    return "<latentthought>\n" + "\n".join(lines) + "\n</latentthought>"
