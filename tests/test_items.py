from j_faithfulness.items import ITEMS
from j_faithfulness.verbalize import words


def test_intermediates_are_hidden():
    for item in ITEMS:
        q = words(item["q"])
        for w in item["intermediate"]:
            assert w.casefold() not in q, (item["slug"], w)
    assert len({i["slug"] for i in ITEMS}) == len(ITEMS)
