from j_faithfulness.items import daily_dilemmas, split


def row(idx, action_type, action, values, situation="Should you?", group="work"):
    return {
        "dilemma_idx": idx,
        "dilemma_situation": situation,
        "action_type": action_type,
        "action": action,
        "negative_consequence": "Something bad happens.",
        "values_aggregated": str(values),
        "topic_group": group,
    }


def test_daily_dilemmas():
    rows = [
        row("55", "not_to_do", "Not report", ["avoidance of conflict"]),
        row("55", "to_do", "Report", ["honesty", "respect for rules", "honesty"]),
    ]
    (item,) = daily_dilemmas(rows)
    assert item.id == "dd-55"
    assert item.q == "Should you? A) Report B) Not report. Answer with A or B only."
    assert item.options == ["A", "B"]
    assert item.actions == {"A": "Report", "B": "Not report"}
    assert item.concepts == {
        "A": ["honesty", "respect for rules"],
        "B": ["avoidance of conflict"],
    }
    assert item.domain == "work"


def test_split_is_fixed_and_disjoint():
    rows = [row(str(i), t, t, []) for i in range(20) for t in ("to_do", "not_to_do")]
    items = daily_dilemmas(rows)
    dev, test = split(items, "dev", 10), split(items, "test", 10)
    assert not {i.id for i in dev} & {i.id for i in test}
    assert split(items, "dev", 3) == dev[:3]
    assert split(list(reversed(items)), "dev", 10) == dev
