"""Items in the shared format (DESIGN.md, Items), and the
DailyDilemmas adapter."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: str
    q: str  # the decision prompt, ending in the forced choice
    options: dict[str, str]  # label → action, in option order
    domain: str
    source: str  # the dataset the item comes from
    source_revision: str


def to_row(item: Item) -> dict:
    """The item as a dataset row. The options become a list in option order,
    so the row's schema doesn't depend on the option labels and rows from
    different sources share one dataset."""
    return {
        "id": item.id,
        "q": item.q,
        "options": [
            {"label": label, "action": action} for label, action in item.options.items()
        ],
        "domain": item.domain,
        "source": item.source,
        "source_revision": item.source_revision,
    }


DAILY_DILEMMAS = "kellycyy/daily_dilemmas"
DAILY_DILEMMAS_REVISION = "372352c4143ec4923c50b4472bd3148a474f7ab8"
DAILY_DILEMMAS_FILE = "dilemma_to_action_to_values_aggregated.csv"


def daily_dilemmas(rows: list[dict]) -> list[Item]:
    """One binary item per dilemma: A = to do, B = not to do. The
    consequences and the annotated values stay out of the item."""
    by_dilemma: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_dilemma.setdefault(row["dilemma_idx"], {})[row["action_type"]] = row
    items = []
    for idx, pair in by_dilemma.items():
        do, dont = pair["to_do"], pair["not_to_do"]
        q = (
            f"{do['dilemma_situation']} A) {do['action']} B) {dont['action']}. "
            "Answer with A or B only."
        )
        items.append(
            Item(
                id=f"dd-{idx}",
                q=q,
                options={"A": do["action"], "B": dont["action"]},
                domain=do["topic_group"],
                source=DAILY_DILEMMAS,
                source_revision=DAILY_DILEMMAS_REVISION,
            )
        )
    return items


def daily_dilemmas_rows(token: str | None = None) -> list[dict]:
    """The rows of the dataset's file, from a pinned revision. ``token``: an
    HF token; None falls back to HF_TOKEN or a cached login."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DAILY_DILEMMAS,
        DAILY_DILEMMAS_FILE,
        repo_type="dataset",
        revision=DAILY_DILEMMAS_REVISION,
        token=token,
    )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_daily_dilemmas(token: str | None = None) -> list[Item]:
    """All 1,360 dilemmas, from a pinned revision of the dataset."""
    return daily_dilemmas(daily_dilemmas_rows(token))


def split(
    items: list[Item], part: str, n: int | None = None, seed: int = 0
) -> list[Item]:
    """The ``"dev"`` or ``"test"`` half of a fixed shuffle (its first ``n``
    items), so dev and test never overlap."""
    shuffled = sorted(items, key=lambda i: i.id)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return {"dev": shuffled[:half], "test": shuffled[half:]}[part][:n]
