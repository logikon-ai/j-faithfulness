import math

import pytest

from j_faithfulness import verbalize as verbalize_module
from j_faithfulness.lens import LatentReadout
from j_faithfulness.verbalize import (
    document,
    echoes,
    grade,
    idf,
    is_english,
    is_latin,
    is_word_like,
    j_and_reversed,
    rows_text,
    term_freqs,
    words,
)


def readout(*units, context="", q_units=None):
    """``units[i][j]``: ranked tokens at unit i, layer j; all units are Q's
    unless ``q_units`` says otherwise."""
    return LatentReadout(
        context=context,
        top=[[list(ranked) for ranked in unit] for unit in units],
        q_units=len(units) if q_units is None else q_units,
    )


def blank(n, q_units=None):
    """A role that carries no words at any of ``n`` units: grading against it
    leaves the scores at TF × IDF."""
    return readout(*[[[]] for _ in range(n)], q_units=q_units)


def tf_idf(r):
    """Per unit: ``{form: (tf, idf)}`` of one readout."""
    return [
        {form: (tf, unit_idf[key]) for key, (tf, form) in unit.items()}
        for unit, unit_idf in zip(term_freqs([r]), idf([r]), strict=True)
    ]


@pytest.fixture
def thresholds(monkeypatch):
    """Sets MIN_TF and MIN_SCORE for one test; each call starts from the
    module's values."""

    def set_thresholds(
        min_tf=verbalize_module.MIN_TF, min_score=verbalize_module.MIN_SCORE
    ):
        monkeypatch.setattr(verbalize_module, "MIN_TF", min_tf)
        monkeypatch.setattr(verbalize_module, "MIN_SCORE", min_score)

    return set_thresholds


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
        assert is_english(w), w
    for w in ["akespeare", "dney", "useum", "hoặc", "atau", "keduanya"]:
        assert not is_english(w), w


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
    r = readout(q0, q1, r_sentence, q_units=2)
    tfs = tf_idf(r)
    # Q's units score against Q's other unit: 1 document.
    assert tfs[0]["Italy"] == pytest.approx((0.5, math.log(2 / 1)))
    # R's sentence scores against both of Q's units, R itself left out.
    assert tfs[2]["Italy"] == pytest.approx((1.0, math.log(3 / 2)))
    assert tfs[2]["the"] == pytest.approx((0.5, math.log(3 / 3)))


def test_tf_idf_single_position_is_plain_tf():
    (pos,) = tf_idf(readout([["Italy", "boot"]]))
    assert pos == {"Italy": (1.0, 1.0), "boot": (0.5, 1.0)}


def test_words():
    assert words("Don't use the euro, Italy!") == {
        "don't",
        "use",
        "the",
        "euro",
        "italy",
    }


def test_term_freqs_pool_by_max():
    a = readout([["fair", "honest"], ["fair"]])
    b = readout([["honest", "Fair"], ["text"]])
    (unit,) = term_freqs([a, b])
    # fair: 1 + 1 in a, 1/2 in b; honest: 1/2 in a, 1 in b.
    assert unit == {"fair": (2, "fair"), "honest": (1, "honest"), "text": (1, "text")}


def test_grade_against_the_other_side():
    decide = readout([["fair", "Answer"], ["fair", "honest"]])
    copy = readout([["Answer", "fair"], ["text"]])
    edit = readout([["honest"], ["text"]])
    # One unit, so IDF is 1. fair: tf 1 + 1 here, 1/2 in copy, 0 in edit;
    # Answer: 1/2 vs 1; honest: 1/2 vs 0 in copy, 1 in edit.
    ours, idfs = term_freqs([decide]), idf([decide])
    (copy_only,) = grade(ours, term_freqs([copy]), idfs)
    assert copy_only == pytest.approx(
        {"fair": (2, 1.5), "Answer": (0.5, -0.5), "honest": (0.5, 0.5)}
    )
    (both,) = grade(ours, term_freqs([copy, edit]), idfs)
    assert both == pytest.approx(
        {"fair": (2, 1.5), "Answer": (0.5, -0.5), "honest": (0.5, -0.5)}
    )


def test_grade_keeps_idf():
    decide = readout([["fair"]], [["fair", "kind"]])
    copy = readout([["text"]], [["kind"]])
    # At unit 1, fair is in the other unit's document: idf log(2/2) = 0.
    # kind: tf 1/2 vs 1 in copy, idf log(2/1).
    _, unit_1 = grade(term_freqs([decide]), term_freqs([copy]), idf([decide]))
    assert unit_1["fair"] == pytest.approx((1, 0))
    assert unit_1["kind"] == pytest.approx((0.5, -0.5 * math.log(2)))


def test_idf_pools_the_q_documents_of_all_readouts():
    a = readout([["Italy"]], [["euro"]], [["Rome"]], q_units=2)
    b = readout([["Italy"]], [["Italy"]], [["euro"]], q_units=2)
    unit_0, unit_1, sentence = idf([a, b])
    # Q's documents: 4, 2 at each unit. At unit 0, the other unit's two
    # documents hold Italy once; at unit 1, Italy twice and euro never.
    assert unit_0 == pytest.approx({"italy": math.log(3 / 2)})
    assert unit_1 == pytest.approx({"euro": math.log(3 / 1), "italy": math.log(3 / 3)})
    # R's sentence scores against all 4: Italy in 3, euro in 1, Rome in 0.
    assert sentence == pytest.approx({"rome": math.log(5 / 1), "euro": math.log(5 / 2)})


def test_readouts_need_the_same_units():
    with pytest.raises(ValueError):
        term_freqs([readout([["x"]]), readout([["x"]], [["y"]])])
    with pytest.raises(ValueError):
        idf([readout([["x"]], [["y"]]), readout([["x"]], [["y"]], q_units=1)])
    with pytest.raises(ValueError):
        j_and_reversed(readout([["x"]]), [readout([["x"]], [["y"]])])


def generic_and_specific():
    """ "the" at every position; one specific word per position."""
    specific = ["Italy", "euro", "Rome", "意大利"]
    return readout(*[[["the", w], ["the", w]] for w in specific])


def test_j_ranks_by_tf_idf_and_drops_generic_words(thresholds):
    thresholds(min_tf=0.5, min_score=0.1)
    j, _ = j_and_reversed(generic_and_specific(), [blank(4)])
    assert j == [["Italy"], ["euro"], ["Rome"]]


def test_j_min_tf_and_min_score(thresholds):
    r = generic_and_specific()
    # Specific words: tf 1/2 + 1/2 = 1, idf log(7/1) ≈ 1.95.
    thresholds(min_tf=1.5, min_score=0.01)
    assert j_and_reversed(r, [blank(4)])[0] == []
    thresholds(min_score=2.0)
    assert j_and_reversed(r, [blank(4)])[0] == []
    thresholds(min_score=1.9)
    assert len(j_and_reversed(r, [blank(4)])[0]) == 3


def test_j_and_reversed_grade_each_side_against_the_other(thresholds):
    decide = readout([["Answer", "fair"]], [["Answer", "kind"]])
    copy = readout([["Answer"]], [["Answer", "text"]])
    # Answer is as strong on both sides: score 0. fair, kind and text: tf
    # 1/2, not on the other side, idf log(2/1): score 0.35.
    thresholds(min_tf=0.5, min_score=0.3)
    assert j_and_reversed(decide, [copy]) == ([["fair"], ["kind"]], [["text"]])
    thresholds(min_tf=0.5, min_score=0.4)
    assert j_and_reversed(decide, [copy]) == ([], [])


def test_reversed_j_pools_two_roles_with_their_joint_idf(thresholds):
    thresholds(min_tf=0.5, min_score=0.1)
    advisor = readout([["harm"]], [["the"]], [["weigh"]], q_units=2)
    # Each advocate carries its side; "harm" in advocate A's Q documents
    # lowers its IDF for both at the third unit.
    pro = readout([["harm"]], [["the"]], [["trust", "honest"]], q_units=2)
    con = readout([["the"]], [["the"]], [["harm"]], q_units=2)
    j, reversed_j = j_and_reversed(advisor, [pro, con])
    # weigh: tf 1, no role carries it, idf log(3/1) over the advisor's Q
    # documents. harm at unit 0 is as strong on both sides: score 0.
    assert j == [["weigh"]]
    # harm: tf 1 (con), idf log(5/2) over the roles' 4 Q documents; trust:
    # tf 1, idf log(5/1); honest: tf 1/2, idf log(5/1).
    assert reversed_j == [["trust", "harm", "honest"]]


def test_both_sides_drop_the_words_of_every_context(thresholds):
    thresholds(min_tf=0.5, min_score=0.1)
    advisor = readout(
        [["Italy", "advisor"]],
        [["euro", "editor"]],
        context="You are an advisor. Does the country shaped like a boot use the euro?",
    )
    editor = readout(
        [["Portugal", "advisor", "boots"]],
        [["Spain", "editor"]],
        context="You are an editor. Does the country shaped like a boot use the euro?",
    )
    # Each side's role word and the shared Q's words drop on both sides.
    assert j_and_reversed(advisor, [editor]) == ([["Italy"]], [["Portugal"], ["Spain"]])


def test_rows_text():
    assert rows_text([["fair", "kind"], ["trust"]]) == "fair, kind.\ntrust."
