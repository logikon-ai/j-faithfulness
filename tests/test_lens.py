import re

from j_faithfulness.formats import Span
from j_faithfulness.lens import (
    read_out,
    sentence_ids,
    skipped_tokens,
    units,
    word_ends,
)


def test_word_ends():
    tokens = ["<|im_start|>", "user", "\n", "The", " guac", "amole", "?", "<|im_end|>"]
    assert word_ends(tokens) == [0, 1, 2, 3, 5, 6, 7]


def test_skipped_tokens():
    tokens = ["<|im_start|>", "user", "\n", "Q", "?"]
    assert skipped_tokens(tokens, "<|im_start|>user\n") == 3
    assert skipped_tokens(tokens, "") == 0
    assert skipped_tokens(tokens, "<|im_start|>us") == 0  # not a token boundary
    assert skipped_tokens(tokens, "<|im_start|>system\n") == 0
    assert skipped_tokens(tokens, "<|im_start|>user\nQ?") == 5


def test_read_out(tiny):
    model, lens = tiny
    context = "the quick brown fox"
    r = read_out(model, lens, context, band=(1, 3), positions=[-1, 4], top_n=5)
    assert r.layers == [1, 2, 3]
    assert r.positions == [len(context), 4]  # + BOS, so -1 is len(context)
    assert len(r.context_tokens) == len(context) + 1
    assert [[len(t) for t in layers] for layers in r.top] == [[5, 5, 5], [5, 5, 5]]
    # Matches the lens at that position and layer.
    logits, _, _ = lens.apply(model, context, layers=[2], positions=[4])
    expected = [model.tokenizer.decode([i]) for i in logits[2][0].topk(5).indices]
    assert r.top[1][1] == expected


def test_read_out_defaults_to_word_ends(tiny):
    model, lens = tiny
    r = read_out(model, lens, "ab cd", band=(4, 4), top_n=5)
    assert r.positions == word_ends(r.context_tokens)


def test_read_out_long_context_is_not_truncated(tiny):
    model, lens = tiny
    # The tiny tokenizer is byte-level: 600 chars is past jlens's default 512.
    r = read_out(model, lens, "a" * 599 + "b", band=(4, 4), positions=[-1], top_n=5)
    assert r.positions == [600]


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
    start, r = context.index(" Q"), Span(context.index(" R"), context.index("\nE"))
    # Tokens: Hi . _Q _ab _cd ? | _R _ef . _gh . | \n E
    got = units(WordTokenizer(), context, start, sentences=r)
    assert got == [[2], [3], [4], [5], [6, 7, 8], [9, 10], [11], [12]]
    assert units(WordTokenizer(), context, start, sentences=r, end=r.start) == [
        [2],
        [3],
        [4],
        [5],
    ]


def test_read_out_unit_takes_the_max_over_its_positions(tiny):
    model, lens = tiny
    context = "the quick brown fox"
    r = read_out(model, lens, context, band=(2, 3), units=[[2, 3, 4], [6]], top_n=5)
    assert r.positions == [4, 6]
    assert r.units == [[2, 3, 4], [6]]
    logits, _, _ = lens.apply(model, context, layers=[3], positions=[2, 3, 4])
    peak = logits[3].log_softmax(-1).amax(0)
    expected = [model.tokenizer.decode([i]) for i in peak.topk(5).indices]
    assert r.top[0][1] == expected
    # A unit of one position reads out as that position.
    single = read_out(model, lens, context, band=(2, 3), positions=[6], top_n=5)
    assert r.top[1] == single.top[0]


def test_sentence_ids():
    tokens = ["She", " trusts", " me", ".\n", "Hon", "esty", "?", " Yes"]
    assert sentence_ids(tokens) == [0, 0, 0, 0, 1, 1, 1, 2]
    tokens = ["Role", ":", " Advisor", ".", "\n", "\n\n", "Task", "."]
    assert sentence_ids(tokens) == [0, 0, 0, 0, 0, 0, 1, 1]
