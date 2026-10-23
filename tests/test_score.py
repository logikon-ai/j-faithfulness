import dataclasses

import pytest
import torch

from j_faithfulness.score import (
    TokenLogprobs,
    assess,
    deltas,
    j_word_gap,
    per_sentence,
    x_logprobs,
)


@pytest.mark.parametrize("chunk", [512, 2])
def test_x_logprobs_match_the_model_logits(tiny, monkeypatch, chunk):
    monkeypatch.setattr("j_faithfulness.score.CHUNK", chunk)
    model, lens = tiny
    text, start = "the context. then x", 13
    out = x_logprobs(model, text, start)
    # Byte tokenizer with BOS: token p + 1 is character p.
    assert out.tokens == [
        model.tokenizer.decode([i]) for i in model.encode(text)[0, 14:]
    ]
    positions = list(range(13, len(text)))
    _, logits, ids = lens.apply(model, text, layers=[], positions=positions)
    expected = logits.log_softmax(-1).gather(-1, ids[0, 14:, None])[:, 0]
    assert torch.allclose(torch.tensor(out.logprobs), expected, atol=1e-5)


def test_deltas():
    based = TokenLogprobs(["a", "b"], [-1.0, -2.0])
    not_based = TokenLogprobs(["a", "b"], [-1.5, -1.0])
    assert deltas(based, not_based) == [0.5, -1.0]
    with pytest.raises(ValueError):
        deltas(based, TokenLogprobs(["a", "c"], [0.0, 0.0]))


def test_per_sentence():
    tokens = ["She", " trusts", " me", ".\n", "Hon", "esty", "?", " Yes"]
    values = [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 2.0, 5.0]
    assert per_sentence(tokens, values) == [2.0] * 4 + [1.0] * 3 + [5.0]


def test_j_word_gap():
    tokens = ["She", " trusts", " me", ".", " Hon", "esty", " wins"]
    values = [0.0, 1.0, 0.0, 0.0, 2.0, 4.0, 0.0]
    # trusts (← trust) and Honesty are J's: mean 7/3; the rest: mean 0.
    assert j_word_gap(tokens, values, {"trust", "honesty"}) == pytest.approx(7 / 3)
    assert j_word_gap(tokens, values, {"nothing"}) is None


@pytest.mark.parametrize(
    ("r", "kind"),
    [("Tell. Yes.", "reasoning"), ("Tell.", "explanation"), (None, "explanation")],
)
def test_assess(tiny_with_offsets, r, kind):
    model, lens = tiny_with_offsets
    row = {
        "id": "dd-1",
        "q": "Tell her? A) Yes B) No. Answer with A or B only.",
        "r": r,
        "d": "A",
        "e": "She wants to know.",
        "metadata": {"model_id": "Qwen/Qwen3.5-4B", "format": "qwen3.5/1"},
    }
    a = assess(model, lens, row, kind, band=(1, 4))
    x = r if kind == "reasoning" else row["e"]
    assert "".join(a.tokens) == model.tokenizer.decode(model.encode(x)[0, 1:].tolist())
    assert len(a.deltas) == len(a.tokens) == len(x)
    assert a.delta == pytest.approx(sum(a.deltas) / len(a.deltas))
    assert a.protocol == (3 if r is None else 4)
    dataclasses.asdict(a)
