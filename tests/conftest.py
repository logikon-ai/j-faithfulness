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
