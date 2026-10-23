import pytest

from j_faithfulness.formats import QWEN35
from j_faithfulness.items import (
    DAILY_DILEMMAS,
    Item,
    daily_dilemmas,
    daily_dilemmas_values,
    from_row,
    row_transcript,
    split,
    to_row,
)


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
    assert item.options == ["A", "B"]
    assert item.actions == {"A": "Report", "B": "Not report"}
    assert item.domain == "work"
    assert item.source == DAILY_DILEMMAS


def test_daily_dilemmas_values():
    rows = [
        {**row("55", "to_do", "Report"), "values_aggregated": "['honesty', 'duty']"},
        {**row("55", "not_to_do", "No"), "values_aggregated": "['loyalty', 'duty']"},
    ]
    assert daily_dilemmas_values(rows) == {"dd-55": ["honesty", "duty", "loyalty"]}


def test_row_round_trip():
    item = Item(
        id="x-1",
        q="Which? 1) Tea 2) Coffee 3) Water",
        options=["1", "2", "3"],
        actions={"1": "Tea", "2": "Coffee", "3": "Water"},
        domain="food",
        source="some/dataset",
        source_revision="abc",
    )
    row = to_row(item)
    assert [o["label"] for o in row["options"]] == ["1", "2", "3"]
    assert from_row({**row, "protocol_3": "..."}) == item


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


def transcript_row(r=None, r_capped=None, fmt="qwen3.5/1"):
    (item,) = daily_dilemmas([row("1", "to_do", "Do"), row("1", "not_to_do", "Don't")])
    return {
        **to_row(item),
        "r": r,
        "d": "A",
        "e": "Because.",
        "metadata": {
            "model_id": "Qwen/Qwen3.5-4B",
            "format": fmt,
            "r_capped": r_capped,
            "e_capped": False,
        },
    }


def test_row_transcript():
    r3, r4 = transcript_row(), transcript_row("Hm...", True)
    q = r3["q"]
    assert row_transcript(r3) == QWEN35.transcript(q, "A", "Because.")
    # The stored R holds the cap mark; it isn't added a second time.
    assert row_transcript(r4) == QWEN35.transcript(q, "A", "Because.", "Hm", True)


def test_row_transcript_rejects_another_format_version():
    with pytest.raises(ValueError):
        row_transcript(transcript_row(fmt="qwen3.5/0"))
