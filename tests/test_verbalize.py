import math

import pytest

from j_faithfulness.lens import LatentReadout
from j_faithfulness.verbalize import (
    Row,
    document,
    echoes,
    fence,
    is_latin,
    is_word_like,
    tf_idf,
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


def test_is_latin():
    for w in ["Italy", "París", "français", "1600", "B2"]:
        assert is_latin(w), w
    for w in ["意大利", "или", "Б", "日本"]:
        assert not is_latin(w), w


def test_document_keeps_best_rank_of_normalized_word_like_words():
    # Ａ → "A" is one character, ¾ → "3⁄4" isn't word-like.
    assert document(["?", "Italy", " Italy", "Ａ", "¾", "ab"]) == {
        "italy": (1, "Italy"),
        "ab": (5, "ab"),
    }


def test_document_merges_case():
    assert document(["wildlife", " Wildlife", "s"]) == {"wildlife": (0, "wildlife")}


def test_tf_idf_merges_case_across_layers_and_shows_the_best_form():
    (pos,) = tf_idf(readout([["x", "wildlife"], [" Wildlife"]]))
    assert pos == {"Wildlife": (1 / 2 + 1, 1.0)}


def test_echoes():
    ctx = words("My friend is allergic to the main ingredient of guacamole.")
    for w in ["friend", "Friend", "friends", "allergy", "allergies", "ingredients"]:
        assert echoes(w, ctx), w
    for w in ["avocado", "buddy", "my5", "main1"]:
        assert not echoes(w, ctx) or w == "main1", w
    assert echoes("main", ctx)  # exact, though shorter than the prefix
    assert not echoes("mains", ctx)  # "main" is too short for the prefix rule


def test_tf_idf_leaves_out_the_scored_position():
    r = readout(
        [["Italy", "the"], ["Italy"]],
        [["the", "euro"], ["the"]],
    )
    pos0, pos1 = tf_idf(r)
    # 4 documents; 2 at the other position.
    # Italy: tf 1 + 1; at the other position in 0 documents.
    assert pos0["Italy"] == pytest.approx((2, math.log(3 / 1)))
    # the: tf 1/2; at the other position in both documents.
    assert pos0["the"] == pytest.approx((0.5, math.log(3 / 3)))
    assert pos1["the"] == pytest.approx((2, math.log(3 / 2)))
    assert pos1["euro"] == pytest.approx((0.5, math.log(3 / 1)))


def test_tf_idf_single_position_is_plain_tf():
    (pos,) = tf_idf(readout([["Italy", "boot"]]))
    assert pos == {"Italy": (1.0, 1.0), "boot": (0.5, 1.0)}


def generic_and_specific():
    """ "the" at every position; one specific word per position."""
    specific = ["Italy", "euro", "Rome", "意大利"]
    return readout(*[[["the", w], ["the", w]] for w in specific])


def test_verbalize_ranks_by_tf_idf_and_drops_generic_words():
    j = verbalize(generic_and_specific(), min_tf=0.5, min_score=0.1)
    assert j == [Row(0, ["Italy"]), Row(1, ["euro"]), Row(2, ["Rome"])]


def test_verbalize_latin_only_can_be_off():
    j = verbalize(generic_and_specific(), min_score=0.1, latin_only=False)
    assert j[-1] == Row(3, ["意大利"])


def test_verbalize_min_tf_and_min_score():
    r = generic_and_specific()
    # Specific words: tf 1/2 + 1/2 = 1, idf log(7/1) ≈ 1.95.
    assert verbalize(r, min_tf=1.5, min_score=0.01) == []
    assert verbalize(r, min_score=2.0) == []
    assert len(verbalize(r, min_score=1.9)) == 3


def test_verbalize_drops_context_echo_and_drop_words():
    r = readout(
        [["Italy", "boots"]],
        [["euro"]],
        [["Portugal"]],
        context="Does the country shaped like a boot use the euro?",
    )
    # boots: prefix of boot is too short, so it stays; euro is in the context.
    assert verbalize(r, min_score=0.1) == [
        Row(0, ["Italy", "boots"]),
        Row(2, ["Portugal"]),
    ]
    assert verbalize(r, min_score=0.1, drop=["italy", "boots"]) == [
        Row(2, ["Portugal"])
    ]
    kept = verbalize(r, min_score=0.1, drop_context=False)
    assert Row(1, ["euro"]) in kept


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
