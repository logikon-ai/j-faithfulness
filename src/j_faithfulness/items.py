"""Items in the shared format (DESIGN.md, Items and adapters), and the
DailyDilemmas adapter."""

from __future__ import annotations

import ast
import csv
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: str
    q: str  # the decision prompt, ending in the forced choice
    options: list[str]  # labels: A, B, …
    actions: dict[str, str]  # label → action
    concepts: dict[str, list[str]]  # label → reference concepts
    domain: str


DAILY_DILEMMAS = "kellycyy/daily_dilemmas"
DAILY_DILEMMAS_REVISION = "372352c4143ec4923c50b4472bd3148a474f7ab8"
DAILY_DILEMMAS_FILE = "dilemma_to_action_to_values_aggregated.csv"


def values(row: dict) -> list[str]:
    """The row's annotated values, each once (the dataset repeats some)."""
    return list(dict.fromkeys(ast.literal_eval(row["values_aggregated"])))


def daily_dilemmas(rows: list[dict]) -> list[Item]:
    """One binary item per dilemma: A = to do, B = not to do, each with the
    values it serves as reference concepts. The consequences stay out of Q."""
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
                options=["A", "B"],
                actions={"A": do["action"], "B": dont["action"]},
                concepts={"A": values(do), "B": values(dont)},
                domain=do["topic_group"],
            )
        )
    return items


def load_daily_dilemmas() -> list[Item]:
    """All 1,360 dilemmas, from a pinned revision of the dataset."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        DAILY_DILEMMAS,
        DAILY_DILEMMAS_FILE,
        repo_type="dataset",
        revision=DAILY_DILEMMAS_REVISION,
    )
    with open(path, newline="", encoding="utf-8") as f:
        return daily_dilemmas(list(csv.DictReader(f)))


def split(
    items: list[Item], part: str, n: int | None = None, seed: int = 0
) -> list[Item]:
    """The ``"dev"`` or ``"test"`` half of a fixed shuffle (its first ``n``
    items), so dev and test never overlap."""
    shuffled = sorted(items, key=lambda i: i.id)
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return {"dev": shuffled[:half], "test": shuffled[half:]}[part][:n]
