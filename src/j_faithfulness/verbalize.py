"""LatentReadout → J (one row of ranked words per position) → fenced text.

Each ranked list of the read-out (one position, one layer) is a document.
At a position, a word scores TF × IDF: TF sums its reciprocal ranks over the
layers, and IDF counts the documents at the other positions that contain it,
so a concept that persists through the layers doesn't raise its own count.

Against baseline read-outs of the same text under other roles (DESIGN.md,
Latent readout, Keyword rows, step 4), J is built one of two ways: graded,
where a word's TF at a unit is reduced by its highest TF at the same unit of
any baseline (:func:`contrast`); or by subtraction, where every word of a
baseline's J is dropped from J wherever it occurs (:func:`verbalize_against`).
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

    IDF is ``log((N + 1) / (df + 1))`` over the documents of Q's units (the
    first ``readout.q_units``; None: all), the scored unit's own left out;
    with no documents left it is 1. IDF from Q alone, so a consideration
    that recurs across R's sentences isn't scored down, and R's denser merged
    documents don't skew it (DESIGN.md, Keyword rows, step 2).
    """
    docs = [[document(ranked) for ranked in layers] for layers in readout.top]
    idf_rows = readout.q_units
    base = docs if idf_rows is None else docs[:idf_rows]
    df = Counter(key for layers in base for d in layers for key in d)
    n = sum(len(layers) for layers in base)
    out = []
    for row, (layers, tfs) in enumerate(zip(docs, term_freqs(readout), strict=True)):
        in_base = idf_rows is None or row < idf_rows
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


def span_words(readout: LatentReadout) -> list[str]:
    """The context token at each read-out position."""
    return [readout.context_tokens[p] for p in readout.positions]


def contrast(
    readout: LatentReadout, *baselines: LatentReadout
) -> list[dict[str, tuple[float, float]]]:
    """Per unit: ``{word: (tf, score)}``, the score ``(tf - tf_base) × idf``.
    ``tf_base`` is the word's highest TF at the same unit of any baseline (0
    where it isn't among a baseline's top tokens): one baseline that carries
    the word explains it away. IDF is the read-out's own (see
    :func:`tf_idf`).

    All read-outs must cover the same words (e.g. the same text under
    different roles).
    """
    for baseline in baselines:
        if span_words(readout) != span_words(baseline):
            raise ValueError("read-outs cover different words")
    base_tfs = [term_freqs(baseline) for baseline in baselines]

    def tf_base(u: int, word: str) -> float:
        return max(
            (b[u].get(word.casefold(), (0.0,))[0] for b in base_tfs), default=0.0
        )

    return [
        {
            word: (tf, (tf - tf_base(u, word)) * idf)
            for word, (tf, idf) in scored.items()
        }
        for u, scored in enumerate(tf_idf(readout))
    ]


def verbalize(
    readout: LatentReadout,
    baselines: list[LatentReadout] | None = None,
    k: int = 5,
    min_tf: float = 1.0,
    min_score: float = 1.0,
    min_zipf: float = 2.5,
) -> J:
    """J: per position, up to ``k`` words by TF × IDF, best first; positions
    left without words are omitted. With ``baselines``, words score by the
    graded :func:`contrast` instead.

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

    if baselines:
        scores = contrast(readout, *baselines)
    else:
        scores = [
            {w: (tf, tf * idf) for w, (tf, idf) in scored.items()}
            for scored in tf_idf(readout)
        ]
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
    *baselines: LatentReadout,
    k: int = 10,
    loose_k: int = 50,
    loose_min_tf: float = 0.5,
) -> J:
    """J of ``readout`` minus every word of each baseline's J (see
    :func:`subtract`), all built loose (``loose_k`` words per position, TF ≥
    ``loose_min_tf``), then up to ``k`` words per position."""
    loose = {"k": loose_k, "min_tf": loose_min_tf}
    seen = [row for baseline in baselines for row in verbalize(baseline, **loose)]
    return subtract(verbalize(readout, **loose), seen, k)


def fence(j: J) -> str:
    """J as a ``<latentthought>`` block, one comma-separated row per line."""
    lines = [", ".join(row.words) + "." for row in j]
    return "<latentthought>\n" + "\n".join(lines) + "\n</latentthought>"
