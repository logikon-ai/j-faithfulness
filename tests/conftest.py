import jlens
import pytest

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


class OffsetByteTokenizer:
    """The tiny model's byte tokenizer, with character offsets for
    :func:`j_faithfulness.lens.units` (ASCII text only)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        encoded = self.tokenizer(text, **kwargs)
        if not return_offsets_mapping:
            return encoded
        ids = encoded.input_ids[0].tolist()
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(ids) - 1)]
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, **kwargs)


@pytest.fixture
def tiny_with_offsets(tiny, monkeypatch):
    model, lens = tiny
    monkeypatch.setattr(model, "tokenizer", OffsetByteTokenizer(model.tokenizer))
    return model, lens
