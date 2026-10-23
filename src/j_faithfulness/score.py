"""The score (DESIGN.md, The score): teacher-forced log-probabilities of X's
tokens under each arm of the probe, and Δ per token."""

from __future__ import annotations

from dataclasses import dataclass

import jlens
import torch
from jlens.hooks import ActivationRecorder

from j_faithfulness import replay
from j_faithfulness.lens import MAX_SEQ_LEN

# Positions unembedded at once: the logits over a 250k vocabulary are the
# large tensor, not the forward pass.
CHUNK = 512


@dataclass
class TokenLogprobs:
    """X's tokens, decoded, and the log-probability of each given everything
    before it."""

    tokens: list[str]
    logprobs: list[float]


@replay.recorded(TokenLogprobs)
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


def sentence_ids(tokens: list[str]) -> list[int]:
    """The sentence each token belongs to; a sentence ends with a token that
    holds ``.``, ``?``, ``!`` or a line break (DESIGN.md, Units)."""
    ids, sentence = [], 0
    for token in tokens:
        ids.append(sentence)
        if any(mark in token for mark in ".?!\n"):
            sentence += 1
    return ids


def per_sentence(tokens: list[str], values: list[float]) -> list[float]:
    """Each token's value replaced by its sentence's mean: the profile."""
    ids = sentence_ids(tokens)
    totals: dict[int, list[float]] = {}
    for sentence, value in zip(ids, values, strict=True):
        totals.setdefault(sentence, []).append(value)
    means = {s: sum(v) / len(v) for s, v in totals.items()}
    return [means[s] for s in ids]
