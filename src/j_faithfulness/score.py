"""The score (DESIGN.md, The score): teacher-forced log-probabilities of X's
tokens under each arm of the probe, Δ per token, and ``assess``, which runs
readout, J, probe and score for one transcript."""

from __future__ import annotations

from dataclasses import dataclass

import jlens
import torch
from jlens.hooks import ActivationRecorder

from j_faithfulness.formats import FORMATS, Kind
from j_faithfulness.items import row_transcript
from j_faithfulness.lens import (
    BAND,
    CHUNK,
    MAX_SEQ_LEN,
    read_span,
    sentence_ids,
    word_ends,
)
from j_faithfulness.probe import probe
from j_faithfulness.verbalize import echoes, fence, verbalize_against


@dataclass
class TokenLogprobs:
    """X's tokens, decoded, and the log-probability of each given everything
    before it."""

    tokens: list[str]
    logprobs: list[float]


@torch.no_grad()
def x_logprobs(model: jlens.LensModel, text: str, start: int) -> TokenLogprobs:
    """Teacher-force ``text`` and return the log-probabilities of its tokens
    from character ``start`` on: X, when ``text`` ends where X ends.

    Raises ValueError if ``start`` isn't a token boundary, since X's tokens
    would then differ between the arms.
    """
    ids = model.encode(text, max_length=MAX_SEQ_LEN)[0]
    n = len(ids)
    if n >= MAX_SEQ_LEN:
        raise ValueError(f"text reaches MAX_SEQ_LEN={MAX_SEQ_LEN} tokens")
    before = model.encode(text[:start], max_length=MAX_SEQ_LEN)[0]
    k = len(before)
    if k == 0 or not torch.equal(ids[:k], before):
        raise ValueError("X doesn't start at a token boundary")

    final_layer = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(ids[None])
        final = recorder.activations[final_layer][0]

    logprobs: list[float] = []
    for first in range(k, n, CHUNK):
        targets = ids[first : first + CHUNK]
        # The residual at position p predicts the token at p + 1.
        logits = model.unembed(final[first - 1 : first - 1 + len(targets)])
        lp = (
            logits.float()
            .log_softmax(-1)
            .gather(-1, targets[:, None].to(logits.device))
        )
        logprobs += lp[:, 0].tolist()
    tokens = [
        model.tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in ids[k:].tolist()
    ]
    return TokenLogprobs(tokens, logprobs)


def deltas(based: TokenLogprobs, not_based: TokenLogprobs) -> list[float]:
    """Δₜ: log P under IS minus log P under is NOT, per token of X."""
    if based.tokens != not_based.tokens:
        raise ValueError("the arms tokenize X differently")
    return [a - b for a, b in zip(based.logprobs, not_based.logprobs, strict=True)]


def per_sentence(tokens: list[str], values: list[float]) -> list[float]:
    """Each token's value replaced by its sentence's mean: the profile."""
    ids = sentence_ids(tokens)
    totals: dict[int, list[float]] = {}
    for sentence, value in zip(ids, values, strict=True):
        totals.setdefault(sentence, []).append(value)
    means = {s: sum(v) / len(v) for s, v in totals.items()}
    return [means[s] for s in ids]


def j_word_gap(
    tokens: list[str], values: list[float], j_words: set[str]
) -> float | None:
    """Mean Δₜ on the tokens of X's words that are J's words or variants of
    them (see :func:`echoes`), minus the mean on X's other tokens; None if
    either set is empty (DESIGN.md, Does the model read J at all?)."""
    on_j, off_j = [], []
    start = 0
    for end in word_ends(tokens):
        word = "".join(tokens[start : end + 1]).strip()
        voiced = word.isalpha() and echoes(word, j_words)
        (on_j if voiced else off_j).extend(values[start : end + 1])
        start = end + 1
    if not on_j or not off_j:
        return None
    return sum(on_j) / len(on_j) - sum(off_j) / len(off_j)


@dataclass
class Assessment:
    """One transcript's test: J's rows, X's tokens with Δₜ, Δ(X) and the
    J-word gap. JSON-serializable via ``dataclasses.asdict``."""

    id: str
    protocol: int
    kind: str
    j: list[list[str]]
    tokens: list[str]
    deltas: list[float]
    delta: float
    j_word_gap: float | None


def assess(
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    row: dict,
    kind: Kind,
    band: tuple[int, int] = BAND,
) -> Assessment:
    """Test one transcript, given as a dataset row: read out J's span under
    the advisor and the typist roles, verbalize J, probe X with J under IS
    and is NOT, and score X under both."""
    fmt = FORMATS[row["metadata"]["model_id"]]
    transcript = row_transcript(row)
    spans = fmt.parse(transcript)
    q, r = row["q"], row["r"]
    advisor = read_span(model, lens, fmt, kind, q, r, band=band)
    typist = read_span(model, lens, fmt, kind, q, r, fmt.COPY_ROLE, band)
    j = verbalize_against(advisor, typist)
    based, not_based = (
        x_logprobs(model, text, x.start)
        for text, x in (
            probe(transcript, spans, kind, fence(j), arm) for arm in (True, False)
        )
    )
    values = deltas(based, not_based)
    j_words = {w.casefold() for j_row in j for w in j_row.words}
    return Assessment(
        id=row["id"],
        protocol=3 if row["r"] is None else 4,
        kind=kind,
        j=[j_row.words for j_row in j],
        tokens=based.tokens,
        deltas=values,
        delta=sum(values) / len(values),
        j_word_gap=j_word_gap(based.tokens, values, j_words),
    )
