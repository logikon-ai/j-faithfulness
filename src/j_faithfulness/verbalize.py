"""Readouts → J and the reversed J: one row of ranked words per unit
(DESIGN.md, Latent readout, Keyword rows).

Each ranked list of a readout (one unit, one layer) is a document. At a
unit, a word's TF sums its reciprocal ranks over the layers, and IDF counts
the documents of Q's other units that contain it, so a concept that persists
through the layers doesn't raise its own count.

J grades the advisor's readout against the contrast's, and the reversed J
the contrast's against the advisor's: a word's TF at a unit is reduced by its
TF at the same unit on the other side (:func:`grade`). Graded, not by
dropping the other side's words: a consideration both sides carry stays
where one weighs it more.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import TYPE_CHECKING

from wordfreq import zipf_frequency

if TYPE_CHECKING:
    # Types only: task.py imports verbalize and stays free of torch.
    from j_faithfulness.lens import LatentReadout

# Words sharing their first ECHO_PREFIX letters with a word of the context
# count as echoing it (allergy, allergies ← allergic), and so do context words
# with one of ECHO_SUFFIXES (funds, funding ← fund), which catches the words
# too short for the prefix rule.
ECHO_PREFIX = 5
ECHO_SUFFIXES = ("s", "es", "ed", "ing", "er", "ers")

# Words per row of J.
K = 5
# A word needs TF ≥ MIN_TF, so rare garbage isn't boosted by its IDF, and a
# score ≥ MIN_SCORE.
MIN_TF = 1.0
MIN_SCORE = 1.0
# The wordfreq frequency (Zipf scale) an English word needs.
MIN_ZIPF = 2.5


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


def is_english(word: str) -> bool:
    """``word`` is an English word or name with wordfreq frequency at least
    MIN_ZIPF. Rules out subword fragments (akespeare, dney) and other
    languages' words (hoặc, atau for "or")."""
    return zipf_frequency(word, "en") >= MIN_ZIPF


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


def readout_term_freqs(readout: LatentReadout) -> list[dict[str, tuple[float, str]]]:
    """Per unit: ``{casefolded word: (tf, form at its best rank)}``, TF
    summing the word's reciprocal ranks over the layers."""
    out = []
    for layers in readout.top:
        tf: Counter[str] = Counter()
        form: dict[str, tuple[int, str]] = {}
        for ranked in layers:
            for key, (rank, word) in document(ranked).items():
                tf[key] += 1 / (rank + 1)
                if key not in form or rank < form[key][0]:
                    form[key] = (rank, word)
        out.append({key: (t, form[key][1]) for key, t in tf.items()})
    return out


def check_aligned(readouts: list[LatentReadout]) -> None:
    if not readouts:
        raise ValueError("no readouts")
    if len({(len(r.top), r.q_units) for r in readouts}) > 1:
        raise ValueError("readouts have different numbers of units or of Q's units")


def term_freqs(readouts: list[LatentReadout]) -> list[dict[str, tuple[float, str]]]:
    """Per unit: ``{casefolded word: (tf, form)}``, the word's highest TF
    over ``readouts`` and its form there. Max, not mean: roles that divide
    an activity, like the two advocates, each carry only their side's
    words, and a mean would halve them."""
    check_aligned(readouts)
    pooled: list[dict[str, tuple[float, str]]] = [{} for _ in readouts[0].top]
    for readout in readouts:
        for unit, tfs in zip(pooled, readout_term_freqs(readout), strict=True):
            for key, (tf, form) in tfs.items():
                if key not in unit or tf > unit[key][0]:
                    unit[key] = (tf, form)
    return pooled


def idf(readouts: list[LatentReadout]) -> list[dict[str, float]]:
    """Per unit: ``{casefolded word: idf}`` for the words of that unit in any
    of ``readouts``.

    IDF is ``log((N + 1) / (df + 1))`` over the documents of Q's units (the
    first ``q_units``) of all ``readouts`` together, the scored unit's own
    left out; with no documents left it is 1. One pool for a side's roles, so
    no rule picks one role's IDF. Q alone, so a consideration that recurs
    across R's sentences isn't scored down, and R's denser merged documents
    don't skew it.
    """
    check_aligned(readouts)
    q_units = readouts[0].q_units
    # docs[u]: the documents of unit u over all readouts and layers.
    docs = [
        [document(ranked) for readout in readouts for ranked in readout.top[u]]
        for u in range(len(readouts[0].top))
    ]
    df = Counter(key for unit in docs[:q_units] for d in unit for key in d)
    n = sum(len(unit) for unit in docs[:q_units])
    out = []
    for u, unit in enumerate(docs):
        found = Counter(key for d in unit for key in d)
        own = found if u < q_units else Counter()
        n_rest = n - len(unit) if u < q_units else n
        out.append(
            {
                key: math.log((n_rest + 1) / (df[key] - own[key] + 1))
                if n_rest
                else 1.0
                for key in found
            }
        )
    return out


def grade(
    tfs: list[dict[str, tuple[float, str]]],
    against: list[dict[str, tuple[float, str]]],
    idfs: list[dict[str, float]],
) -> list[dict[str, tuple[float, float]]]:
    """Per unit: ``{form: (tf, score)}``, the score ``(tf - tf_against) ×
    idf``, ``tf_against`` 0 where the word isn't among the other side's
    top tokens."""
    return [
        {
            form: (tf, (tf - other.get(key, (0.0,))[0]) * unit_idf[key])
            for key, (tf, form) in unit.items()
        }
        for unit, other, unit_idf in zip(tfs, against, idfs, strict=True)
    ]


def keyword_rows(
    scored: list[dict[str, tuple[float, float]]], echo_words: set[str]
) -> list[list[str]]:
    """Per unit, up to K words, best first; units left without words are
    omitted.

    A word needs TF ≥ MIN_TF and a score ≥ MIN_SCORE. Left out are words
    that echo ``echo_words`` (at a word of Q the lens partly predicts Q's
    next word), words in other scripts (mostly translations of English
    ones), and words that aren't English (see :func:`is_english`).
    """

    def keep(w: str) -> bool:
        return not echoes(w, echo_words) and is_latin(w) and is_english(w)

    rows = []
    for unit in scored:
        ranked = sorted(
            (
                (score, w)
                for w, (tf, score) in unit.items()
                if tf >= MIN_TF and score >= MIN_SCORE and keep(w)
            ),
            key=lambda sw: -sw[0],
        )
        if ranked:
            rows.append([w for _, w in ranked[:K]])
    return rows


def j_and_reversed(
    advisor: LatentReadout, contrast: list[LatentReadout]
) -> tuple[list[list[str]], list[list[str]]]:
    """J and the reversed J of one case, from the advisor's readout and the
    contrast roles' (aligned, see :func:`j_faithfulness.lens.read_case`).

    J grades the advisor's TF against the contrast's highest TF over its
    roles, with the advisor's IDF. The reversed J grades that highest TF
    against the advisor's, with IDF over all roles' readouts. Both sides
    drop the words of every context read, so of Q, R and every role line:
    no side's own words tell J from the reversed J.
    """
    check_aligned([advisor, *contrast])
    advisor_tfs = term_freqs([advisor])
    contrast_tfs = term_freqs(contrast)
    echo_words = set().union(*(words(r.context) for r in [advisor, *contrast]))
    j = keyword_rows(grade(advisor_tfs, contrast_tfs, idf([advisor])), echo_words)
    reversed_j = keyword_rows(
        grade(contrast_tfs, advisor_tfs, idf(contrast)), echo_words
    )
    return j, reversed_j


def rows_text(rows: list[list[str]]) -> str:
    """A readout as the model sees it: one row per line."""
    return "\n".join(", ".join(row) + "." for row in rows)
