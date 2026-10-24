"""LatentReadout → J: one row of ranked words per unit.

Each ranked list of the read-out (one unit, one layer) is a document. At a
unit, a word's TF sums its reciprocal ranks over the layers, and IDF counts
the documents of Q's other units that contain it, so a concept that persists
through the layers doesn't raise its own count.

Graded against read-outs of the same text under other roles (DESIGN.md,
Latent readout), a word's TF at a unit is reduced by its highest TF at the
same unit of any of them (:func:`grade`). Graded, not by dropping their
words: a consideration both roles carry stays where one weighs it more.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from wordfreq import zipf_frequency

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


def term_freqs(readout: LatentReadout) -> list[dict[str, tuple[float, str]]]:
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


def tf_idf(readout: LatentReadout) -> list[dict[str, tuple[float, float]]]:
    """Per unit: ``{word: (tf, idf)}``, each word in the form it has at its
    best rank there.

    IDF is ``log((N + 1) / (df + 1))`` over the documents of Q's units (the
    first ``readout.q_units``), the scored unit's own left out;
    with no documents left it is 1. IDF from Q alone, so a consideration
    that recurs across R's sentences isn't scored down, and R's denser merged
    documents don't skew it (DESIGN.md, Keyword rows, step 2).
    """
    docs = [[document(ranked) for ranked in layers] for layers in readout.top]
    base = docs[: readout.q_units]
    df = Counter(key for layers in base for d in layers for key in d)
    n = sum(len(layers) for layers in base)
    out = []
    for row, (layers, tfs) in enumerate(zip(docs, term_freqs(readout), strict=True)):
        in_base = row < readout.q_units
        own = Counter(key for d in layers for key in d) if in_base else Counter()
        n_rest = n - len(layers) if in_base else n
        out.append(
            {
                word: (
                    t,
                    math.log((n_rest + 1) / (df[key] - own[key] + 1))
                    if n_rest
                    else 1.0,
                )
                for key, (t, word) in tfs.items()
            }
        )
    return out


def grade(
    readout: LatentReadout, *against: LatentReadout
) -> list[dict[str, tuple[float, float]]]:
    """Per unit: ``{word: (tf, score)}``, the score ``(tf - tf_against) ×
    idf``. ``tf_against`` is the word's highest TF at the same unit of any
    read-out in ``against`` (0 where it isn't among its top tokens): max, not
    mean, so one role that carries the word explains it away. IDF is the
    read-out's own (see :func:`tf_idf`).

    The read-outs must have their units in the same order, one to one. Their
    words may differ: the copy editor reads R in the user turn, so its
    template sits elsewhere (:func:`j_faithfulness.lens.align_to_advisor`).
    """
    for other in against:
        if len(other.top) != len(readout.top):
            raise ValueError("read-outs have different numbers of units")
    against_tfs = [term_freqs(other) for other in against]

    def tf_against(u: int, word: str) -> float:
        return max(
            (a[u].get(word.casefold(), (0.0,))[0] for a in against_tfs), default=0.0
        )

    return [
        {
            word: (tf, (tf - tf_against(u, word)) * idf)
            for word, (tf, idf) in scored.items()
        }
        for u, scored in enumerate(tf_idf(readout))
    ]


def verbalize(readout: LatentReadout, against: list[LatentReadout]) -> list[list[str]]:
    """J: per unit, up to K words by :func:`grade`, best first; units left
    without words are omitted.

    A word needs TF ≥ MIN_TF and a score ≥ MIN_SCORE. Left out are words of
    the context and their variants (at a word of Q the lens partly predicts
    Q's next word), words in other scripts (mostly translations of English
    ones), and words that aren't English (see :func:`is_english`).
    """
    context_words = words(readout.context)

    def keep(w: str) -> bool:
        return not echoes(w, context_words) and is_latin(w) and is_english(w)

    j = []
    for scored in grade(readout, *against):
        ranked = sorted(
            (
                (score, w)
                for w, (tf, score) in scored.items()
                if tf >= MIN_TF and score >= MIN_SCORE and keep(w)
            ),
            key=lambda sw: -sw[0],
        )
        if ranked:
            j.append([w for _, w in ranked[:K]])
    return j
