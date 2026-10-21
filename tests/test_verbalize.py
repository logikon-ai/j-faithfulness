from j_faithfulness.lens import LatentReadout
from j_faithfulness.verbalize import fence, is_word_like, verbalize, words


def readout(*rows):
    return LatentReadout(
        layers=list(range(len(rows))),
        tokens=[list(r) for r in rows],
        token_ids=[list(range(len(r))) for r in rows],
        n_context_tokens=1,
    )


def test_is_word_like():
    for t in [" Neither", "A", "don't", "e-mail", "形状", " 2024"]:
        assert is_word_like(t), t
    for t in ["", " ", "\n\n", " **", "?", "-shaped", "'?", "<think>", "<|im_end|>"]:
        assert not is_word_like(t), t


def test_verbalize_filters_merges_and_truncates():
    r = readout(
        [" **", " Neither", "\n\n", "Neither", " Both", "A", " A", "?", "B"],
        [" library", "bus", " Library"],
    )
    assert verbalize(r, k=3) == [
        ["Neither", "Both", "A"],
        ["library", "bus", "Library"],
    ]
    assert verbalize(r, k=10)[0] == ["Neither", "Both", "A", "B"]


def test_verbalize_drop():
    r = readout([" library", " Library", " books", " bus"])
    q = "Should the town fund a library or a bus line?"
    assert verbalize(r, drop=words(q)) == [["books"]]


def test_fence():
    assert (
        fence([["a", "b"], [], ["c"]]) == "<latentthought>\na, b.\nc.\n</latentthought>"
    )
