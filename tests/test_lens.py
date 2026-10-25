import re
from types import SimpleNamespace

import jlens
import pytest
import torch

from j_faithfulness import lens as lens_module
from j_faithfulness.contrasts import for_options, load_contrast, role_context
from j_faithfulness.formats import QWEN35
from j_faithfulness.lens import (
    Segments,
    align,
    batches,
    read_case,
    read_out,
    segments,
    sentence_ids,
    tokenize,
    top_per_unit,
    units,
    word_ends,
)

from .tiny import TinyDecoder


def test_word_ends():
    tokens = ["<|im_start|>", "user", "\n", "The", " guac", "amole", "?", "<|im_end|>"]
    assert word_ends(tokens) == [0, 1, 2, 3, 5, 6, 7]


def test_read_out(tiny):
    model, lens = tiny
    context = "the quick brown fox"
    r = read_out(model, lens, context, band=(1, 3), units=[[19], [4]], q_units=1)
    assert r.context == context
    assert r.q_units == 1
    # Three layers per unit; the tiny vocabulary is smaller than TOP_N.
    assert [[len(t) for t in layers] for layers in r.top] == [[32] * 3, [32] * 3]
    # Matches the lens at that position and layer.
    logits, _, _ = lens.apply(model, context, layers=[2], positions=[4])
    expected = [model.tokenizer.decode([i]) for i in logits[2][0].topk(32).indices]
    assert r.top[1][1] == expected


def test_read_out_long_context_is_not_truncated(tiny):
    model, lens = tiny
    # The tiny tokenizer is byte-level: 600 chars is past jlens's default 512.
    r = read_out(model, lens, "a" * 599 + "b", band=(4, 4), units=[[600]], q_units=1)
    logits, _, _ = lens.apply(
        model, "a" * 599 + "b", layers=[4], positions=[600], max_seq_len=1024
    )
    expected = [model.tokenizer.decode([i]) for i in logits[4][0].topk(32).indices]
    assert r.top[0][0] == expected


class WordTokenizer:
    """One token per word (with its leading space) or punctuation mark."""

    def __call__(self, text, return_offsets_mapping=False):
        found = list(re.finditer(r" ?\w+|[^\w ]", text))
        self.vocab = {i: m.group() for i, m in enumerate(found)}
        return {
            "input_ids": list(self.vocab),
            "offset_mapping": [m.span() for m in found],
        }

    def decode(self, ids, **_kw):
        return "".join(self.vocab[i] for i in ids)


def test_units():
    context = "Hi. Q ab cd? R ef. gh.\nE"
    start, r = context.index(" Q"), range(context.index(" R"), context.index("\nE"))
    tokens, offsets = tokenize(WordTokenizer(), context)
    # Tokens: Hi . _Q _ab _cd ? | _R _ef . _gh . | \n E
    got = units(tokens, offsets, start, sentences=r)
    assert got == [[2], [3], [4], [5], [6, 7, 8], [9, 10], [11], [12]]
    assert units(tokens, offsets, start) == [[p] for p in range(2, 13)]


def test_segments():
    context = "Hi. Q ab? <a> R ef. gh. </a> D"
    # Tokens: Hi . _Q _ab ? _< a > | _R _ef . _gh . | _< / a > _D
    got = segments(WordTokenizer(), context, " Q ab?", " R ef. gh.")
    assert got == Segments(
        q=[[2], [3], [4]],
        between=[[5], [6], [7]],
        r=[[8, 9, 10], [11, 12]],
        after=[[13], [14], [15], [16], [17]],
    )


def test_segments_without_r():
    context = "Hi. Q ab? <a> D"
    # Tokens: Hi . _Q _ab ? _< a > _D
    got = segments(WordTokenizer(), context, " Q ab?", None)
    assert got == Segments(
        q=[[2], [3], [4]], between=[], r=[], after=[[5], [6], [7], [8]]
    )


def test_align_with_r():
    # Explanation, thinking on: Q and R pair; between gets the role's last
    # unit before R, after gets its last unit.
    advisor = Segments([[0]], [[1], [2]], [[3]], [[4], [5]])
    role = Segments([[10]], [[11]], [[12]], [[13], [14], [15]])
    assert align(advisor, role) == [[10], [11], [11], [12], [15], [15]]
    # A role with nothing between Q and R: its last unit before R is Q's.
    assert align(advisor, role._replace(between=[])) == (
        [[10], [10], [10], [12], [15], [15]]
    )


def test_align_without_r():
    # Reasoning, and explanation with thinking off: every advisor unit after
    # Q gets the role's last unit, however many units each side has there.
    advisor = Segments([[0], [1]], [], [], [[2], [3], [4]])
    role = Segments([[10], [11]], [], [], [[12], [13]])
    assert align(advisor, role) == [[10], [11], [13], [13], [13]]
    assert align(advisor, role._replace(after=[])) == [[10], [11], [11], [11], [11]]


def test_align_needs_q_and_r_to_split_alike():
    advisor = Segments([[0]], [[1]], [[2]], [[3]])
    role = Segments([[10]], [[11]], [[12]], [[13]])
    with pytest.raises(ValueError):
        align(advisor, role._replace(r=[]))
    with pytest.raises(ValueError):
        align(advisor, role._replace(q=[[10], [11]]))


def test_read_out_unit_takes_the_max_over_its_positions(tiny):
    model, lens = tiny
    context = "the quick brown fox"
    r = read_out(model, lens, context, band=(2, 3), units=[[2, 3, 4], [6]], q_units=2)
    logits, _, _ = lens.apply(model, context, layers=[3], positions=[2, 3, 4])
    peak = logits[3].log_softmax(-1).amax(0)
    expected = [model.tokenizer.decode([i]) for i in peak.topk(32).indices]
    assert r.top[0][1] == expected
    # A unit of one position reads out as that position.
    single = read_out(model, lens, context, band=(2, 3), units=[[6]], q_units=1)
    assert r.top[1] == single.top[0]


def test_sentence_ids():
    tokens = ["She", " trusts", " me", ".\n", "Hon", "esty", "?", " Yes"]
    assert sentence_ids(tokens) == [0, 0, 0, 0, 1, 1, 1, 2]
    tokens = ["Role", ":", " Advisor", ".", "\n", "\n\n", "Task", "."]
    assert sentence_ids(tokens) == [0, 0, 0, 0, 0, 0, 1, 1]


def top_per_unit_reference(states, read_units, unembed, top_n, chunk=512):
    """The per-unit loop that :func:`top_per_unit` replaced."""
    ranked = []
    for unit in read_units:
        best = None
        for c in range(0, len(unit), chunk):
            logits = unembed(states[unit[c : c + chunk]]).float()
            peak = logits.log_softmax(-1).amax(0)
            best = peak if best is None else torch.maximum(best, peak)
        ranked.append(best.topk(min(top_n, len(best))).indices.tolist())
    return ranked


# Multi-position and single-position units, one overlapping another, and one
# repeating a position.
READ_UNITS = [[1, 2, 3], [4], [5, 6, 7, 8, 9], [10], [9, 10, 11], [12, 12], [0]]


@pytest.mark.parametrize("chunk, unit_block", [(2048, 2048), (3, 2), (1, 1)])
def test_top_per_unit_matches_the_per_unit_loop(tiny, monkeypatch, chunk, unit_block):
    # Small CHUNK and UNIT_BLOCK make chunks span units and units span chunks.
    monkeypatch.setattr(lens_module, "CHUNK", chunk)
    monkeypatch.setattr(lens_module, "UNIT_BLOCK", unit_block)
    model, lens = tiny
    torch.manual_seed(0)
    states = lens.transport(torch.randn(13, model.d_model), 2)
    for top_n in (5, 100):
        got = top_per_unit(states, READ_UNITS, model.unembed, top_n)
        assert got == top_per_unit_reference(states, READ_UNITS, model.unembed, top_n)


def test_top_per_unit_rejects_an_empty_unit(tiny):
    model, _ = tiny
    with pytest.raises(ValueError):
        top_per_unit(torch.randn(3, model.d_model), [[0], []], model.unembed, 5)


class CharTokenizer:
    """One token per ASCII character, its code point, after a BOS: the tiny
    model's tokenizer with offsets and tokens that decode to their text."""

    bos_token_id = 0

    def __call__(self, text, *, max_length=None, return_offsets_mapping=False, **_kw):
        ids = [self.bos_token_id] + [ord(c) for c in text]
        ids = ids[:max_length]
        if return_offsets_mapping:
            offsets = [(0, 0)] + [(i, i + 1) for i in range(len(ids) - 1)]
            return {"input_ids": ids, "offset_mapping": offsets}
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids, **_kw):
        return "".join(chr(int(i)) for i in ids)


@pytest.fixture(scope="module")
def tiny_chars():
    model = TinyDecoder(n_layers=6, d_model=8, vocab_size=128)
    model.tokenizer = CharTokenizer()
    lens = jlens.fit(
        model,
        ["abcdefghij " * 5],
        source_layers=[0, 1, 2, 3, 4],
        dim_batch=4,
        max_seq_len=64,
    )
    return model, lens


CASE_Q = "Should I tell my friend?\n\nA. Tell her.\nB. Don't tell her."
CASE_R = "She trusts me. But it would hurt her!\n\nSo what matters more?\nHonesty."
CASES = {
    "reasoning": ("reasoning", CASE_R),
    "explanation, thinking off": ("explanation", None),
    "explanation, thinking on": ("explanation", CASE_R),
}


@pytest.mark.parametrize("contrast", ["editor", "advocates"])
@pytest.mark.parametrize("case", CASES)
def test_read_case(tiny_chars, contrast, case):
    model, lens = tiny_chars
    kind, r = CASES[case]
    roles = load_contrast(contrast)
    advisor, readouts = read_case(model, lens, (1, 3), roles, CASE_Q, kind, r, "AB")

    read_r = r if kind == "explanation" else None
    advisor_context = (
        QWEN35.reasoning_prompt(CASE_Q)
        if kind == "reasoning"
        else QWEN35.decision_prompt(CASE_Q, r)
    )
    assert advisor.context == advisor_context
    assert [c.context for c in readouts] == [
        role_context(role, CASE_Q, read_r) for role in for_options(roles, ["A", "B"])
    ]

    advisor_segments = segments(model.tokenizer, advisor_context, CASE_Q, read_r)
    q_units = len(advisor_segments.q)
    n_units = sum(len(part) for part in advisor_segments)
    assert advisor.q_units == q_units
    assert len(advisor.top) == n_units
    for readout in readouts:
        assert readout.q_units == q_units
        assert len(readout.top) == n_units
        # Every advisor unit after the last of Q and R reads the role's last.
        after = readout.top[n_units - len(advisor_segments.after) :]
        assert all(unit == after[-1] for unit in after)
        # Every advisor unit between Q and R reads one unit of the role's.
        between = readout.top[q_units : q_units + len(advisor_segments.between)]
        assert all(unit == between[0] for unit in between)


def test_batches_shortest_first_within_both_caps():
    lengths = [5, 1, 3, 2, 4]
    assert batches(lengths, batch_size=2, batch_tokens=100) == [[1, 3], [2, 4], [0]]
    # Padded size: 3 rows of 3 tokens fit in 9, a fourth of 4 tokens would not.
    assert batches(lengths, batch_size=10, batch_tokens=9) == [[1, 3, 2], [4], [0]]
    # A prompt longer than the token cap gets a batch of its own.
    assert batches([20, 1], batch_size=10, batch_tokens=9) == [[1], [0]]
    assert batches([]) == []
