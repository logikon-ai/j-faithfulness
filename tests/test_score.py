import dataclasses

import pytest
import torch

from j_faithfulness.score import (
    Assessment,
    TokenLogprobs,
    against_control,
    assess,
    deltas,
    j_word_gap,
    j_word_gaps,
    mismatched,
    per_sentence,
    spread,
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


def assessment(kind, protocol, delta, gap, item="dd-1", j=()):
    return Assessment(item, protocol, kind, list(j), [], [], delta, gap)


def test_model_level_checks():
    assessments = [
        assessment("reasoning", 4, 0.1, 0.2),
        assessment("reasoning", 4, -0.1, None),
        assessment("explanation", 3, 0.3, 0.1),
    ]
    reasoning = spread(assessments)["reasoning, protocol 4"]
    assert reasoning == pytest.approx(
        {"n": 2, "mean": 0.0, "std": 0.1, "min": -0.1, "max": 0.1}
    )
    gaps = j_word_gaps(assessments)
    assert gaps["reasoning, protocol 4"] == pytest.approx(
        {"mean": 0.2, "n": 1, "dropped": 1}
    )
    assert gaps["explanation, protocol 3"]["mean"] == pytest.approx(0.1)


def test_mismatched_takes_the_next_items_j_in_the_same_case():
    own = [
        assessment("reasoning", 4, 0, 0, "dd-1", [["a"]]),
        assessment("explanation", 3, 0, 0, "dd-1", [["x"]]),
        assessment("reasoning", 4, 0, 0, "dd-2", [["b"]]),
        assessment("reasoning", 4, 0, 0, "dd-3", [["c"]]),
    ]
    assert mismatched(own) == [[["b"]], [["x"]], [["c"]], [["a"]]]


def test_against_control_pairs_by_item():
    own = [
        assessment("reasoning", 4, 0.0, 0.3, "dd-1"),
        assessment("reasoning", 4, -0.1, None, "dd-2"),
        assessment("reasoning", 4, 0.1, 0.2, "dd-3"),
    ]
    control = [
        assessment("reasoning", 4, -0.1, 0.0, "dd-3"),
        assessment("reasoning", 4, -0.2, 0.1, "dd-1"),
        assessment("reasoning", 4, -0.2, 0.1, "dd-2"),
    ]
    stats = against_control(own, control)["reasoning, protocol 4"]
    # Δ differences 0.2, 0.1, 0.2; gap differences 0.2, 0.2 (dd-2 drops out).
    assert stats["delta own − control"] == pytest.approx(0.5 / 3)
    assert stats["j_word_gap own − control"] == pytest.approx(0.2)
    assert stats["j_word_gap n"] == 2


def test_assess_with_a_given_j_reads_nothing(tiny_with_offsets, monkeypatch):
    model, lens = tiny_with_offsets
    monkeypatch.setattr(
        "j_faithfulness.score.read_span", lambda *a, **k: pytest.fail("read out")
    )
    row = {
        "id": "dd-1",
        "q": "Tell her? A) Yes B) No. Answer with A or B only.",
        "r": None,
        "d": "A",
        "e": "She wants to know.",
        "metadata": {"model_id": "Qwen/Qwen3.5-4B", "format": "qwen3.5/1"},
    }
    a = assess(model, lens, row, "explanation", j=[["want", "know"]])
    assert a.j == [["want", "know"]]
