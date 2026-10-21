import random

from j_faithfulness.ablations import (
    distinct_words,
    drop,
    drop_random,
    drop_top,
    present,
    replace_top,
    swap,
    swap_random,
    truncate,
)
from j_faithfulness.verbalize import Row


def j():
    return [
        Row(3, ["Italy", "boots", "heel"], [5.0, 2.0, 1.0]),
        Row(7, ["Europe", "italy"], [3.0, 4.0]),
        Row(9, ["Commonwealth"], [0.5]),
    ]


def words(jj):
    return [row.words for row in jj]


def test_distinct_words_by_salience():
    assert distinct_words(j()) == ["Italy", "Europe", "boots", "heel", "Commonwealth"]


def test_swap_everywhere_casefolded():
    assert words(swap(j(), {"Italy": "Norway"})) == [
        ["Norway", "boots", "heel"],
        ["Europe", "Norway"],
        ["Commonwealth"],
    ]


def test_swap_does_not_repeat_a_word_in_a_row():
    assert words(swap(j(), {"Europe": "Italy"}))[1] == ["Italy"]


def test_swap_random_puts_the_foils_elsewhere():
    mapping = {"Italy": "Norway", "Rome": "Oslo"}  # Rome isn't in J
    assert present(j(), mapping) == ["Italy"]
    for seed in range(10):
        out = swap_random(j(), mapping, random.Random(seed))
        flat = [w for row in out for w in row.words]
        assert flat.count("Norway") == 1
        assert "Italy" in flat and "Oslo" not in flat


def test_swap_random_respects_exclude():
    out = swap_random(
        j(), {"Italy": "Norway"}, random.Random(0), exclude=["boots", "heel", "Europe"]
    )
    assert words(out)[-1] == ["Norway"]


def test_drop_and_drop_top_omit_empty_rows():
    assert words(drop(j(), ["commonwealth"])) == words(j())[:2]
    assert words(drop_top(j(), 2)) == [["boots", "heel"], ["Commonwealth"]]


def test_drop_random():
    out = drop_random(j(), 2, random.Random(0), exclude=["Italy"])
    assert sum(len(w) for w in words(out)) == 6 - 2
    assert "Italy" in words(out)[0]


def test_replace_top_and_truncate():
    assert words(replace_top(j(), 1, ["Norway"]))[1] == ["Europe", "Norway"]
    assert words(truncate(j(), 2)) == words(j())[:2]
