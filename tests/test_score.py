import pytest
import torch

from j_faithfulness.score import (
    TokenLogprobs,
    deltas,
    per_sentence,
    sentence_ids,
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


def test_sentences():
    tokens = ["She", " trusts", " me", ".\n", "Hon", "esty", "?", " Yes"]
    assert sentence_ids(tokens) == [0, 0, 0, 0, 1, 1, 1, 2]
    values = [1.0, 2.0, 3.0, 2.0, 0.0, 1.0, 2.0, 5.0]
    assert per_sentence(tokens, values) == [2.0] * 4 + [1.0] * 3 + [5.0]
    tokens = ["Role", ":", " Advisor", ".", "\n", "\n\n", "Task", "."]
    assert sentence_ids(tokens) == [0, 0, 0, 0, 0, 0, 1, 1]
