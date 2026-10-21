from j_faithfulness.lens import LatentReadout
from j_faithfulness.verbalize import (
    Row,
    fence,
    is_word_like,
    scores,
    verbalize,
    words,
)


def readout(*positions, context=""):
    """``positions[i][j]``: ranked tokens at position i, layer j."""
    return LatentReadout(
        context=context,
        context_tokens=[],
        positions=list(range(len(positions))),
        layers=list(range(len(positions[0]))),
        top=[[list(ranked) for ranked in p] for p in positions],
    )


def test_is_word_like():
    for t in [" Neither", "A", "don't", "e-mail", "形状", " 2024"]:
        assert is_word_like(t), t
    for t in ["", " ", "\n\n", " **", "?", "-shaped", "'?", "<think>", "<|im_end|>"]:
        assert not is_word_like(t), t


def test_scores_sum_reciprocal_ranks_over_layers():
    # Layer 0: Italy at rank 1, " Italy" at 2 (same word: best rank counts).
    # Layer 1: Italy at rank 0.
    s = scores([["?", "Italy", " Italy", "Ａ"], [" Italy", "boot"]])
    assert s == {"Italy": 1 / 2 + 1, "A": 1 / 4, "boot": 1 / 2}


def test_scores_check_the_normalized_word():
    assert scores([["¾", "Italy"]]) == {"Italy": 1 / 2}  # ¾ → "3⁄4"


def test_verbalize_thresholds_and_ranks():
    r = readout(
        [["Italy", "x", "y", "z", "w", "v", "u", "t", "s", "r", "boot"]],
        [["?", "?"]],
    )
    # boot scores 1/11 < 0.1; the second position has no words left.
    assert verbalize(r, k=3) == [Row(0, ["Italy", "x", "y"])]
    assert verbalize(r, k=3, min_score=0) == [Row(0, ["Italy", "x", "y"])]
    assert verbalize(r, k=20, min_score=0)[0].words[-1] == "boot"


def test_verbalize_drops_context_words_and_generic_words():
    r = readout(
        [["the", "Italy"]],
        [["the", "euro"]],
        [["the", "boot"]],
        context="Does the country shaped like a boot use the euro?",
    )
    # "the" is generic (all positions) and in the context; boot, euro are in it.
    assert verbalize(r) == [Row(0, ["Italy"])]
    kept = verbalize(r, drop_context=False, max_share=1.0)
    assert [row.words for row in kept] == [
        ["the", "Italy"],
        ["the", "euro"],
        ["the", "boot"],
    ]
    assert verbalize(r, drop=["italy"]) == []


def test_single_position_words_are_never_generic():
    assert verbalize(readout([["Italy"]])) == [Row(0, ["Italy"])]


def test_words():
    assert words("Don't use the euro, Italy!") == {
        "don't",
        "use",
        "the",
        "euro",
        "italy",
    }


def test_fence():
    j = [Row(3, ["a", "b"]), Row(7, ["c"])]
    assert fence(j) == "<latentthought>\na, b.\nc.\n</latentthought>"
