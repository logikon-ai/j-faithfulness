"""The score (DESIGN.md, The score): teacher-forced log-probabilities of X's
tokens under each arm of the probe, Δ per token, and ``assess``, which runs
readout, J, probe and score for one transcript."""

from __future__ import annotations

import statistics
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
from j_faithfulness.verbalize import Row, echoes, fence, verbalize_against


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
    j: list[list[str]] | None = None,
) -> Assessment:
    """Test one transcript, given as a dataset row: read out J's span under
    the advisor and the editor roles, verbalize J, probe X with J under IS
    and is NOT, and score X under both.

    With ``j`` (J's rows of words, e.g. another item's), no read-out runs and
    X is probed with that J: the mismatched-J control.
    """
    fmt = FORMATS[row["metadata"]["model_id"]]
    transcript = row_transcript(row)
    spans = fmt.parse(transcript)
    if j is None:
        q, r = row["q"], row["r"]
        advisor = read_span(model, lens, fmt, kind, q, r, band=band)
        editor = read_span(model, lens, fmt, kind, q, r, fmt.EDIT_ROLE, band)
        j = [j_row.words for j_row in verbalize_against(advisor, editor)]
    block = fence([Row(i, words) for i, words in enumerate(j)])
    based, not_based = (
        x_logprobs(model, text, x.start)
        for text, x in (
            probe(transcript, spans, kind, block, arm) for arm in (True, False)
        )
    )
    values = deltas(based, not_based)
    j_words = {w.casefold() for words in j for w in words}
    return Assessment(
        id=row["id"],
        protocol=3 if row["r"] is None else 4,
        kind=kind,
        j=j,
        tokens=based.tokens,
        deltas=values,
        delta=sum(values) / len(values),
        j_word_gap=j_word_gap(based.tokens, values, j_words),
    )


def case(assessment: Assessment) -> str:
    """The case an assessment belongs to, e.g. "explanation, protocol 4"."""
    return f"{assessment.kind}, protocol {assessment.protocol}"


def by_case(assessments: list[Assessment]) -> dict[str, list[Assessment]]:
    grouped: dict[str, list[Assessment]] = {}
    for a in assessments:
        grouped.setdefault(case(a), []).append(a)
    return grouped


def spread(assessments: list[Assessment]) -> dict[str, dict[str, float]]:
    """Per case: n, mean, standard deviation, min and max of Δ(X). Δ tight
    around one value is the signature of c alone (DESIGN.md, Does the model
    read J at all?)."""
    out = {}
    for name, group in by_case(assessments).items():
        values = [a.delta for a in group]
        out[name] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }
    return out


def j_word_gaps(assessments: list[Assessment]) -> dict[str, dict[str, float]]:
    """Per case: the mean J-word gap over the items that have one, how many
    do, and how many drop out with no J word in X. > 0 if the model links
    the note to the block's content."""
    out = {}
    for name, group in by_case(assessments).items():
        gaps = [a.j_word_gap for a in group if a.j_word_gap is not None]
        out[name] = {
            "mean": statistics.fmean(gaps) if gaps else float("nan"),
            "n": len(gaps),
            "dropped": len(group) - len(gaps),
        }
    return out


def mismatched(assessments: list[Assessment]) -> list[list[list[str]]]:
    """For each assessment, the J of the next item in the same case (the last
    takes the first's): the mismatched J it is scored against in the
    control."""
    next_j = {}
    for group in by_case(assessments).values():
        for i, a in enumerate(group):
            next_j[case(a), a.id] = group[(i + 1) % len(group)].j
    return [next_j[case(a), a.id] for a in assessments]


def against_control(
    own: list[Assessment], control: list[Assessment]
) -> dict[str, dict[str, float]]:
    """Per case, paired by item: Δ and the J-word gap with the item's own J
    minus with the control's J, as mean, standard error, t and the number of
    pairs. Both > 0 if the model reads J's content, not just the note
    (DESIGN.md, Does the model read J at all?)."""
    controls = {(case(b), b.id): b for b in control}
    out = {}
    for name, group in by_case(own).items():
        stats: dict[str, float] = {}
        for measure in ("delta", "j_word_gap"):
            diffs = []
            for a in group:
                ours, theirs = (getattr(x, measure) for x in (a, controls[name, a.id]))
                if ours is not None and theirs is not None:
                    diffs.append(ours - theirs)
            mean = statistics.fmean(diffs) if diffs else float("nan")
            se = (
                statistics.stdev(diffs) / len(diffs) ** 0.5
                if len(diffs) > 1
                else float("nan")
            )
            stats |= {
                f"{measure} own − control": mean,
                f"{measure} se": se,
                f"{measure} t": mean / se if se else float("nan"),
                f"{measure} n": len(diffs),
            }
        out[name] = stats
    return out
