"""Conditions derived from own J (DESIGN.md, Conditions and expected
contrasts). Every edit returns a new J; rows left without words are omitted.
Words are matched casefolded.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

from j_faithfulness.verbalize import J, Row


def _edit(j: J, fn: Callable[[str, float], str | None]) -> J:
    """Apply ``fn(word, score)`` to every word: a replacement, or None to
    drop it. A replacement already in the row is not repeated."""
    out = []
    for row in j:
        words, scores = [], []
        for w, sc in zip(row.words, row.scores or [0.0] * len(row.words), strict=True):
            new = fn(w, sc)
            if new is not None and new.casefold() not in {x.casefold() for x in words}:
                words.append(new)
                scores.append(sc)
        if words:
            out.append(Row(row.position, words, scores))
    return out


def distinct_words(j: J) -> list[str]:
    """J's words, each once (casefolded), by salience: best TF × IDF first."""
    best: dict[str, tuple[float, str]] = {}
    for row in j:
        for w, sc in zip(row.words, row.scores or [0.0] * len(row.words), strict=True):
            key = w.casefold()
            if key not in best or sc > best[key][0]:
                best[key] = (sc, w)
    return [w for _, w in sorted(best.values(), key=lambda sw: -sw[0])]


def swap(j: J, mapping: dict[str, str]) -> J:
    """Replace each word in ``mapping`` with its foil wherever it occurs."""
    foils = {w.casefold(): f for w, f in mapping.items()}
    return _edit(j, lambda w, _: foils.get(w.casefold(), w))


def present(j: J, words: Iterable[str]) -> list[str]:
    """Those of ``words`` that occur in J."""
    in_j = {w.casefold() for w in distinct_words(j)}
    return [w for w in words if w.casefold() in in_j]


def swap_random(
    j: J, mapping: dict[str, str], rng: random.Random, exclude: Iterable[str] = ()
) -> J:
    """Baseline for :func:`swap`: the foils of the ``mapping`` words that occur
    in J replace as many randomly chosen other words (not in ``exclude``)
    instead, so the foils appear in J either way."""
    foils = [mapping[w] for w in present(j, mapping)]
    skip = {w.casefold() for w in [*mapping, *exclude]}
    candidates = [w for w in distinct_words(j) if w.casefold() not in skip]
    chosen = rng.sample(candidates, min(len(foils), len(candidates)))
    return swap(j, dict(zip(chosen, foils, strict=False)))


def drop(j: J, words: Iterable[str]) -> J:
    """Remove ``words`` wherever they occur."""
    gone = {w.casefold() for w in words}
    return _edit(j, lambda w, _: None if w.casefold() in gone else w)


def drop_top(j: J, n: int) -> J:
    """Remove the ``n`` most salient words."""
    return drop(j, distinct_words(j)[:n])


def drop_random(j: J, n: int, rng: random.Random, exclude: Iterable[str] = ()) -> J:
    """Baseline for :func:`drop_top`: remove ``n`` random words (not in
    ``exclude``)."""
    skip = {w.casefold() for w in exclude}
    candidates = [w for w in distinct_words(j) if w.casefold() not in skip]
    return drop(j, rng.sample(candidates, min(n, len(candidates))))


def replace_top(j: J, n: int, foils: list[str]) -> J:
    """Replace the ``n`` most salient words with ``foils``, in order."""
    return swap(j, dict(zip(distinct_words(j)[:n], foils, strict=False)))


def truncate(j: J, s: int) -> J:
    """The first ``s`` rows: the thoughts up to that word of the context."""
    return [Row(r.position, list(r.words), list(r.scores)) for r in j[:s]]
