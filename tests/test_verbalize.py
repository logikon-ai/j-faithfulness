import math

import pytest

from j_faithfulness.lens import LatentReadout
from j_faithfulness.verbalize import (
    Row,
    contrast,
    document,
    echoes,
    fence,
    in_context,
    is_english,
    is_latin,
    is_word_like,
    subtract,
    tf_idf,
    touches,
    verbalize,
    words,
)


def readout(*positions, context="", tokens=None):
    """``positions[i][j]``: ranked tokens at position i, layer j; ``tokens``
    the context token at each position."""
    return LatentReadout(
        context=context,
        context_tokens=tokens or [f"w{i}" for i in range(len(positions))],
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
    for w in ["avocado", "buddy", "mainly", "isle"]:
        assert not echoes(w, ctx), w
    # Short context words: exact, or with a suffix.
    for w in ["main", "mains", "tos"]:
        assert echoes(w, ctx), w


def test_echoes_suffixes_of_short_words():
    ctx = words("Should the town fund a bus line?")
    for w in ["funds", "funded", "funding", "buses", "towns", "lines"]:
        assert echoes(w, ctx), w
    for w in ["fun", "busy", "township"]:
        assert not echoes(w, ctx), w


def test_is_english():
    for w in ["Italy", "Shakespeare", "Nairobi", "venom", "avocado", "bees"]:
        assert is_english(w, 2.5), w
    for w in ["akespeare", "dney", "useum", "hoặc", "atau", "keduanya"]:
        assert not is_english(w, 2.5), w


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


def test_verbalize_min_tf_and_min_score():
    r = generic_and_specific()
    # Specific words: tf 1/2 + 1/2 = 1, idf log(7/1) ≈ 1.95.
    assert verbalize(r, min_tf=1.5, min_score=0.01) == []
    assert verbalize(r, min_score=2.0) == []
    assert len(verbalize(r, min_score=1.9)) == 3


def test_verbalize_drops_context_echo():
    r = readout(
        [["Italy", "boots"]],
        [["euro"]],
        [["Portugal"]],
        context="Does the country shaped like a boot use the euro?",
    )
    # boots echoes boot (suffix); euro is in the context.
    assert verbalize(r, min_tf=0.5, min_score=0.1) == [
        Row(0, ["Italy"]),
        Row(2, ["Portugal"]),
    ]


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


def test_contrast_diff_and_ratio():
    decide = readout([["fair", "Answer"], ["fair", "honest"]])
    copy = readout([["Answer", "fair"], ["text"]])
    # fair: tf 1 + 1 here, 1/2 in copy; Answer: 1/2 vs 1; honest: 1/2 vs 0.
    (diff,) = contrast(decide, copy)
    assert diff == pytest.approx(
        {"fair": (2, 1.5), "Answer": (0.5, -0.5), "honest": (0.5, 0.5)}
    )
    (ratio,) = contrast(decide, copy, "ratio", smoothing=0.5)
    assert ratio["fair"] == pytest.approx((2, math.log(2.5 / 1)))
    assert ratio["honest"] == pytest.approx((0.5, math.log(1 / 0.5)))


def test_contrast_needs_the_same_words():
    a = readout([["x"]], tokens=["Q"])
    b = readout([["x"]], tokens=["P"])
    with pytest.raises(ValueError):
        contrast(a, b)


def test_verbalize_against_a_baseline():
    decide = readout([["Answer", "fair"]], [["Answer", "kind"]])
    copy = readout([["Answer"]], [["Answer"]])
    # Answer is as strong in copy: diff 0. fair and kind: tf 1/2, diff 1/2.
    assert verbalize(decide, copy, min_tf=0.5, min_score=0.5) == [
        Row(0, ["fair"]),
        Row(1, ["kind"]),
    ]
    assert verbalize(decide, copy, min_tf=0.5, min_score=0.6) == []


def test_touches():
    j = [Row(0, ["honest", "others"]), Row(1, ["career"])]
    assert touches(j, "Honesty")
    assert touches(j, "respect for others")
    assert touches(j, "Career advancement")
    assert not touches(j, "Loyalty")
    # Stopwords don't match.
    assert not touches([Row(0, ["for"])], "respect for autonomy")


def test_in_context():
    q = "Should I tell my friend the honest truth about the loan?"
    assert in_context("Honesty", q)
    assert in_context("Friendship", q)
    assert not in_context("Loyalty", q)
    assert not in_context("Respect for others", "Should I go for it?")


def test_subtract_drops_baseline_words_anywhere():
    j = [
        Row(0, ["Girlfriend", "fair", "kind"]),
        Row(1, ["marriage"]),
        Row(2, ["a", "b", "c"]),
    ]
    baseline = [Row(5, ["girlfriend", "marriage"])]
    assert subtract(j, baseline, k=2) == [Row(0, ["fair", "kind"]), Row(2, ["a", "b"])]
