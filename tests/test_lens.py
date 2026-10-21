import jlens
import pytest

from j_faithfulness.lens import band_layers, read_at

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


def test_band_layers():
    assert band_layers((20, 26)) == [20, 22, 24, 26]
    assert band_layers((20, 25)) == [20, 22, 24]
    assert band_layers((1, 3), stride=1) == [1, 2, 3]


def test_read_at(tiny):
    model, lens = tiny
    context = "the quick brown fox"
    r = read_at(model, lens, context, band=(0, 4), top_n=5)
    assert r.layers == [0, 2, 4]
    assert r.n_context_tokens == len(context) + 1  # + BOS
    assert [len(t) for t in r.tokens] == [5, 5, 5]
    assert r.tokens[0] == [model.tokenizer.decode([i]) for i in r.token_ids[0]]
    # Matches the lens at the last position.
    logits, _, _ = lens.apply(model, context, layers=[2], positions=[-1])
    assert r.token_ids[1] == logits[2][0].topk(5).indices.tolist()


def test_read_at_long_context_is_not_truncated(tiny):
    model, lens = tiny
    # The tiny tokenizer is byte-level: 600 chars is past jlens's default 512.
    r = read_at(model, lens, "a" * 599 + "b", band=(4, 4), top_n=5)
    assert r.n_context_tokens == 601
