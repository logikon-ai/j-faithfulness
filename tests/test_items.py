from j_faithfulness.items import DAILY_DILEMMAS, Item, daily_dilemmas, split, to_row


def row(idx, action_type, action, situation="Should you?", group="work"):
    return {
        "dilemma_idx": idx,
        "dilemma_situation": situation,
        "action_type": action_type,
        "action": action,
        "negative_consequence": "Something bad happens.",
        "topic_group": group,
    }


def test_daily_dilemmas():
    rows = [
        row("55", "not_to_do", "Not report"),
        row("55", "to_do", "Report"),
    ]
    (item,) = daily_dilemmas(rows)
    assert item.id == "dd-55"
    assert item.q == "Should you? A) Report B) Not report. Answer with A or B only."
    assert item.options == {"A": "Report", "B": "Not report"}
    assert item.domain == "work"
    assert item.source == DAILY_DILEMMAS


def test_row_keeps_option_order():
    item = Item(
        id="x-1",
        q="Which? 2) Coffee 1) Tea 3) Water",
        options={"2": "Coffee", "1": "Tea", "3": "Water"},
        domain="food",
        source="some/dataset",
        source_revision="abc",
    )
    assert to_row(item)["options"] == [
        {"label": "2", "action": "Coffee"},
        {"label": "1", "action": "Tea"},
        {"label": "3", "action": "Water"},
    ]


def test_row_schema_does_not_depend_on_labels():
    (item,) = daily_dilemmas([row("1", "to_do", "Do"), row("1", "not_to_do", "Don't")])
    r = to_row(item)
    assert set(r) == {"id", "q", "options", "domain", "source", "source_revision"}
    assert all(set(o) == {"label", "action"} for o in r["options"])


def test_split_is_fixed_and_disjoint():
    rows = [row(str(i), t, t) for i in range(20) for t in ("to_do", "not_to_do")]
    items = daily_dilemmas(rows)
    dev, test = split(items, "dev", 10), split(items, "test", 10)
    assert not {i.id for i in dev} & {i.id for i in test}
    assert split(items, "dev", 3) == dev[:3]
    assert split(list(reversed(items)), "dev", 10) == dev
