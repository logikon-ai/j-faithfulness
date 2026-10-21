import jlens
import pytest

from j_faithfulness.lens import read_out, skipped_tokens, word_ends

from .tiny import TinyDecoder


@pytest.fixture(scope="module")
def tiny():
    model = TinyDecoder(n_layers=6, d_model=8)
    lens = jlens.fit(
        model,
        ["abcdefghij " * 5],
        source_layers=[0, 1, 2, 3, 4],
        dim_batch=4,
        max_seq_len=64,
    )
    return model, lens


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


def test_in_band(tiny):
    model, lens = tiny
    r = read_out(model, lens, "ab cd", band=(0, 4), top_n=5)
    b = r.in_band(1, 2)
    assert b.layers == [1, 2]
    assert b.top == [ranked[1:3] for ranked in r.top]
    assert b.positions == r.positions
