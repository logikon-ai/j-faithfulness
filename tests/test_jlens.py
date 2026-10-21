"""jlens is installed and its fit → apply round-trip works on the tiny CPU model."""

import jlens

from .tiny import TinyDecoder


def test_fit_and_apply_tiny():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = jlens.fit(
        model,
        ["abcdefghij " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    lens_logits, model_logits, input_ids = lens.apply(
        model, "the quick brown fox", layers=[1, 2], positions=[-1]
    )
    assert set(lens_logits) == {1, 2}
    assert lens_logits[1].shape == (1, 32)
    assert model_logits.shape == (1, 32)
    assert input_ids.shape[-1] == len("the quick brown fox") + 1  # + BOS
