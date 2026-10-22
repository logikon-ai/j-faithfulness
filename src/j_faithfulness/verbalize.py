"""LatentReadout → J (one row of ranked words per position) → fenced text.

Each ranked list of the read-out (one position, one layer) is a document.
At a position, a word scores TF × IDF: TF sums its reciprocal ranks over the
layers, and IDF counts the documents at the other positions that contain it,
so a concept that persists through the layers doesn't raise its own count.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from wordfreq import zipf_frequency

from j_faithfulness.lens import LatentReadout

# Words sharing their first ECHO_PREFIX letters with a word of the context
# count as echoing it (allergy, allergies ← allergic), and so do context words
# with one of ECHO_SUFFIXES (funds, funding ← fund), which catches the words
# too short for the prefix rule.
ECHO_PREFIX = 5
ECHO_SUFFIXES = ("s", "es", "ed", "ing", "er", "ers")


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


def is_latin(word: str) -> bool:
    """Every letter is Latin (accents allowed); digits and marks pass."""
    return all(not ch.isalpha() or "LATIN" in unicodedata.name(ch, "") for ch in word)


def normalize(token: str) -> str:
    """Stripped and NFKC-normalized, so ``" A"``, ``"A"`` and the fullwidth
    ``"Ａ"`` are one word."""
    return unicodedata.normalize("NFKC", token.strip())


def words(text: str) -> set[str]:
    """Casefolded words of ``text``."""
    return {w.casefold() for w in re.findall(r"\w+(?:['’-]\w+)*", text)}


def echoes(word: str, context_words: set[str]) -> bool:
    """``word`` is a context word, shares its first ECHO_PREFIX letters with
    one, or is one with a suffix from ECHO_SUFFIXES."""
    w = word.casefold()
    if w in context_words:
        return True
    if any(
        w.endswith(sfx) and w[: -len(sfx)] in context_words for sfx in ECHO_SUFFIXES
    ):
        return True
    return len(w) >= ECHO_PREFIX and any(
        c[:ECHO_PREFIX] == w[:ECHO_PREFIX]
        for c in context_words
        if len(c) >= ECHO_PREFIX
    )


def is_english(word: str, min_zipf: float) -> bool:
    """``word`` is an English word or name with wordfreq frequency (Zipf
    scale) at least ``min_zipf``. Rules out subword fragments (akespeare,
    dney) and other languages' words (hoặc, atau for "or")."""
    return zipf_frequency(word, "en") >= min_zipf


def document(ranked: list[str]) -> dict[str, tuple[int, str]]:
    """One ranked list as a document: each word-like word of two or more
    characters at its best rank, keyed casefolded (Wildlife, wildlife), with
    the form found there."""
    best: dict[str, tuple[int, str]] = {}
    for rank, token in enumerate(ranked):
        word = normalize(token)
        if len(word) >= 2 and is_word_like(word):
            best.setdefault(word.casefold(), (rank, word))
    return best


def tf_idf(readout: LatentReadout) -> list[dict[str, tuple[float, float]]]:
    """Per position: ``{word: (tf, idf)}``, each word in the form it has at
    its best rank there.

    IDF is ``log((N + 1) / (df + 1))`` over the documents at the other
    positions; with no other positions it is 1.
    """
    docs = [[document(ranked) for ranked in layers] for layers in readout.top]
    df = Counter(key for layers in docs for d in layers for key in d)
    n = sum(len(layers) for layers in docs)
    out = []
    for layers in docs:
        own = Counter(key for d in layers for key in d)
        tf: Counter[str] = Counter()
        form: dict[str, tuple[int, str]] = {}
        for d in layers:
            for key, (rank, word) in d.items():
                tf[key] += 1 / (rank + 1)
                if key not in form or rank < form[key][0]:
                    form[key] = (rank, word)
        n_rest = n - len(layers)
        out.append(
            {
                form[key][1]: (
                    t,
                    math.log((n_rest + 1) / (df[key] - own[key] + 1))
                    if n_rest
                    else 1.0,
                )
                for key, t in tf.items()
            }
        )
    return out


def verbalize(
    readout: LatentReadout,
    k: int = 5,
    min_tf: float = 1.0,
    min_score: float = 1.0,
    min_zipf: float = 2.5,
) -> J:
    """J: per position, up to ``k`` words by TF × IDF, best first; positions
    left without words are omitted.

    A word needs TF ≥ ``min_tf`` (so rare garbage isn't boosted by its IDF)
    and TF × IDF ≥ ``min_score``. Left out are words of the context and their
    variants (at a word of Q the lens partly predicts Q's next word), words in
    other scripts (mostly translations of English ones), and words that aren't
    English (see :func:`is_english`). The defaults were tuned on the one-hop
    validation items (DESIGN.md).
    """
    context_words = words(readout.context)

    def keep(w: str) -> bool:
        return not echoes(w, context_words) and is_latin(w) and is_english(w, min_zipf)

    j = []
    for position, scored in zip(readout.positions, tf_idf(readout), strict=True):
        ranked = sorted(
            (
                (tf * idf, w)
                for w, (tf, idf) in scored.items()
                if tf >= min_tf and tf * idf >= min_score and keep(w)
            ),
            key=lambda sw: -sw[0],
        )
        if ranked:
            j.append(Row(position, [w for _, w in ranked[:k]]))
    return j


def fence(j: J) -> str:
    """J as a ``<latentthought>`` block, one comma-separated row per line."""
    lines = [", ".join(row.words) + "." for row in j]
    return "<latentthought>\n" + "\n".join(lines) + "\n</latentthought>"
