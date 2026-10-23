import math

import pytest

from j_faithfulness.lens import LatentReadout
from j_faithfulness.verbalize import (
    Row,
    contrast,
    document,
    echoes,
    fence,
    is_english,
    is_latin,
    is_word_like,
    tf_idf,
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


def test_tf_idf_over_q_alone():
    q0 = [["the", "Italy"]]
    q1 = [["the", "euro"]]
    r_sentence = [["Italy", "the"]]
    r = readout(q0, q1, r_sentence)
    r.q_units = 2
    tfs = tf_idf(r)
    # Q's units score against Q's other unit: 1 document.
    assert tfs[0]["Italy"] == pytest.approx((0.5, math.log(2 / 1)))
    # R's sentence scores against both of Q's units, R itself left out.
    assert tfs[2]["Italy"] == pytest.approx((1.0, math.log(3 / 2)))
    assert tfs[2]["the"] == pytest.approx((0.5, math.log(3 / 3)))


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


def test_contrast_is_graded_by_the_strongest_baseline():
    decide = readout([["fair", "Answer"], ["fair", "honest"]])
    copy = readout([["Answer", "fair"], ["text"]])
    edit = readout([["honest"], ["text"]])
    # One unit, so IDF is 1. fair: tf 1 + 1 here, 1/2 in copy, 0 in edit;
    # Answer: 1/2 vs 1; honest: 1/2 vs 0 in copy, 1 in edit.
    (copy_only,) = contrast(decide, copy)
    assert copy_only == pytest.approx(
        {"fair": (2, 1.5), "Answer": (0.5, -0.5), "honest": (0.5, 0.5)}
    )
    (both,) = contrast(decide, copy, edit)
    assert both == pytest.approx(
        {"fair": (2, 1.5), "Answer": (0.5, -0.5), "honest": (0.5, -0.5)}
    )


def test_contrast_keeps_idf():
    decide = readout([["fair"]], [["fair", "kind"]])
    copy = readout([["text"]], [["kind"]])
    # At unit 1, fair is in the other unit's document: idf log(2/2) = 0.
    # kind: tf 1/2 vs 1 in copy, idf log(2/1).
    _, unit_1 = contrast(decide, copy)
    assert unit_1["fair"] == pytest.approx((1, 0))
    assert unit_1["kind"] == pytest.approx((0.5, -0.5 * math.log(2)))


def test_contrast_needs_the_same_words():
    a = readout([["x"]], tokens=["Q"])
    b = readout([["x"]], tokens=["P"])
    with pytest.raises(ValueError):
        contrast(a, b)


def test_verbalize_graded_against_a_baseline():
    decide = readout([["Answer", "fair"]], [["Answer", "kind"]])
    copy = readout([["Answer"]], [["Answer"]])
    # Answer is as strong in copy: score 0. fair and kind: tf 1/2, not in
    # copy, idf log(2/1): score 0.35.
    assert verbalize(decide, [copy], min_tf=0.5, min_score=0.3) == [
        Row(0, ["fair"]),
        Row(1, ["kind"]),
    ]
    assert verbalize(decide, [copy], min_tf=0.5, min_score=0.4) == []
