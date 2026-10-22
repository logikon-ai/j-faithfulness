"""LatentReadout → J (one row of ranked words per position) → fenced text.

Each ranked list of the read-out (one position, one layer) is a document.
At a position, a word scores TF × IDF: TF sums its reciprocal ranks over the
layers, and IDF counts the documents at the other positions that contain it,
so a concept that persists through the layers doesn't raise its own count.

Against a baseline read-out of the same Q under another task (the contrastive
read-out, PLAN.md M1), a word instead scores its TF here against its TF at the
same word of the baseline.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Literal

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


def term_freqs(readout: LatentReadout) -> list[dict[str, tuple[float, str]]]:
    """Per position: ``{casefolded word: (tf, form at its best rank)}``, TF
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
    """Per position: ``{word: (tf, idf)}``, each word in the form it has at
    its best rank there.

    IDF is ``log((N + 1) / (df + 1))`` over the documents at the other
    positions; with no other positions it is 1.
    """
    docs = [[document(ranked) for ranked in layers] for layers in readout.top]
    df = Counter(key for layers in docs for d in layers for key in d)
    n = sum(len(layers) for layers in docs)
    out = []
    for layers, tfs in zip(docs, term_freqs(readout), strict=True):
        own = Counter(key for d in layers for key in d)
        n_rest = n - len(layers)
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


def span_words(readout: LatentReadout) -> list[str]:
    """The context token at each read-out position."""
    return [readout.context_tokens[p] for p in readout.positions]


def contrast(
    readout: LatentReadout,
    baseline: LatentReadout,
    measure: Literal["diff", "ratio"] = "diff",
    smoothing: float = 0.1,
) -> list[dict[str, tuple[float, float]]]:
    """Per position: ``{word: (tf, score)}``, scoring TF here against TF at
    the same word of ``baseline`` (0 where the word isn't in its top tokens):
    their difference, or ``log((tf + smoothing) / (tf_base + smoothing))``.

    Both read-outs must cover the same words (e.g. Q under two task
    prefixes).
    """
    if span_words(readout) != span_words(baseline):
        raise ValueError("read-outs cover different words")
    out = []
    for tfs, base in zip(term_freqs(readout), term_freqs(baseline), strict=True):
        scored = {}
        for key, (t, word) in tfs.items():
            b = base.get(key, (0.0, ""))[0]
            if measure == "diff":
                scored[word] = (t, t - b)
            else:
                scored[word] = (t, math.log((t + smoothing) / (b + smoothing)))
        out.append(scored)
    return out


def verbalize(
    readout: LatentReadout,
    baseline: LatentReadout | None = None,
    measure: Literal["diff", "ratio"] = "diff",
    k: int = 5,
    min_tf: float = 1.0,
    min_score: float = 1.0,
    min_zipf: float = 2.5,
) -> J:
    """J: per position, up to ``k`` words by TF × IDF, best first; positions
    left without words are omitted. With a ``baseline``, words score by
    :func:`contrast` instead.

    A word needs TF ≥ ``min_tf`` (so rare garbage isn't boosted by its IDF)
    and a score ≥ ``min_score``. Left out are words of the context and their
    variants (at a word of Q the lens partly predicts Q's next word), words in
    other scripts (mostly translations of English ones), and words that aren't
    English (see :func:`is_english`). The defaults were tuned on the one-hop
    validation items (DESIGN.md).
    """
    context_words = words(readout.context)

    def keep(w: str) -> bool:
        return not echoes(w, context_words) and is_latin(w) and is_english(w, min_zipf)

    if baseline is None:
        scores = [
            {w: (tf, tf * idf) for w, (tf, idf) in scored.items()}
            for scored in tf_idf(readout)
        ]
    else:
        scores = contrast(readout, baseline, measure)
    j = []
    for position, scored in zip(readout.positions, scores, strict=True):
        ranked = sorted(
            (
                (score, w)
                for w, (tf, score) in scored.items()
                if tf >= min_tf and score >= min_score and keep(w)
            ),
            key=lambda sw: -sw[0],
        )
        if ranked:
            j.append(Row(position, [w for _, w in ranked[:k]]))
    return j


def subtract(j: J, baseline: J, k: int) -> J:
    """J without any word that occurs anywhere in ``baseline``, or a variant
    of one (see :func:`echoes`: couple ← couples), then up to ``k`` words per
    position; positions left without words are omitted."""
    seen = {w.casefold() for row in baseline for w in row.words}
    out = []
    for row in j:
        kept = [w for w in row.words if not echoes(w, seen)][:k]
        if kept:
            out.append(Row(row.position, kept))
    return out


def verbalize_against(
    readout: LatentReadout,
    baseline: LatentReadout,
    k: int = 10,
    loose_k: int = 50,
    loose_min_tf: float = 0.5,
) -> J:
    """J of ``readout`` minus every word of ``baseline``'s J (see
    :func:`subtract`), both built loose (``loose_k`` words per position, TF ≥
    ``loose_min_tf``), then up to ``k`` words per position."""
    loose = {"k": loose_k, "min_tf": loose_min_tf}
    return subtract(verbalize(readout, **loose), verbalize(baseline, **loose), k)


# Left out when matching reference concepts against J ("respect for others").
CONCEPT_STOPWORDS = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}


def concept_words(concept: str) -> set[str]:
    return words(concept) - CONCEPT_STOPWORDS


def touches(j: J, concept: str) -> bool:
    """A word of J is a word of ``concept`` or a variant of one (see
    :func:`echoes`): honest touches Honesty."""
    cw = concept_words(concept)
    return any(echoes(w, cw) for row in j for w in row.words)


def in_context(concept: str, context: str) -> bool:
    """A word of ``concept`` echoes ``context``. Such a concept can't show in J,
    which drops context echo."""
    cw = words(context)
    return any(echoes(w, cw) for w in concept_words(concept))


def fence(j: J) -> str:
    """J as a ``<latentthought>`` block, one comma-separated row per line."""
    lines = [", ".join(row.words) + "." for row in j]
    return "<latentthought>\n" + "\n".join(lines) + "\n</latentthought>"
