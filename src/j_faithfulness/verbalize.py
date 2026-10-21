"""LatentReadout → J (one row of ranked words per layer) → fenced text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from j_faithfulness.lens import LatentReadout

J = list[list[str]]


def is_word_like(token: str) -> bool:
    """jlens's ``mask_display`` rule: after stripping whitespace, only
    alphanumerics, with ' - ’ allowed inside; no special tokens."""
    s = token.strip()
    if not s or "<|" in s or (s.startswith("<") and s.endswith(">")):
        return False
    return all(
        ch.isalnum() or (0 < i < len(s) - 1 and ch in "'-’") for i, ch in enumerate(s)
    )


def words(text: str) -> set[str]:
    """Casefolded words of ``text``, e.g. of Q for ``verbalize(drop=...)``."""
    return {w.casefold() for w in re.findall(r"\w+(?:['’-]\w+)*", text)}


def verbalize(readout: LatentReadout, k: int = 10, drop: Iterable[str] = ()) -> J:
    """The top ``k`` word-like tokens of each layer, in rank order.

    Tokens are stripped and NFKC-normalized, so ``" A"``, ``"A"`` and the
    fullwidth ``"Ａ"`` become one word, kept at the best rank. Words in ``drop``
    (compared casefolded) are left out.
    """
    dropped = {w.casefold() for w in drop}
    rows = []
    for tokens in readout.tokens:
        row: list[str] = []
        for token in tokens:
            word = unicodedata.normalize("NFKC", token.strip())
            if (
                is_word_like(word)
                and word not in row
                and word.casefold() not in dropped
            ):
                row.append(word)
                if len(row) == k:
                    break
        rows.append(row)
    return rows


def fence(j: J) -> str:
    """J as a ``<latentthought>`` block, one comma-separated row per line."""
    lines = [", ".join(row) + "." for row in j if row]
    return "<latentthought>\n" + "\n".join(lines) + "\n</latentthought>"
