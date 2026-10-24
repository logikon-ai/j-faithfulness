"""Contrast roles as data (DESIGN.md, Latent readout): a contrast is a YAML
file listing its roles. The built-in contrasts ship in ``contrasts/`` and load
by name; one's own loads by path."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml

from j_faithfulness.formats import QWEN35

PLACEHOLDER = "{text}"


@dataclass(frozen=True)
class Role:
    """A contrast role. ``template`` is the text of its user turn, with
    PLACEHOLDER once for what it reads. ``reads_r``: R goes into the user
    turn after Q (``text``) or into the role's own think block (``think``)."""

    name: str
    template: str
    reads_r: Literal["text", "think"]

    def __post_init__(self) -> None:
        if self.template.count(PLACEHOLDER) != 1:
            raise ValueError(f"{self.name}: the template needs {PLACEHOLDER} once")
        if self.reads_r not in ("text", "think"):
            raise ValueError(f"{self.name}: reads_r must be 'text' or 'think'")


def load_contrast(name_or_path: str | Path) -> list[Role]:
    """The roles of a built-in contrast, by name, or of a contrast file, by
    path."""
    builtin = files("j_faithfulness").joinpath(f"contrasts/{name_or_path}.yaml")
    source = builtin if builtin.is_file() else Path(name_or_path)
    roles = [Role(**d) for d in yaml.safe_load(source.read_text())]
    if not roles:
        raise ValueError(f"contrast {name_or_path!r} has no roles")
    return roles


def contrast_name(name_or_path: str | Path) -> str:
    """A contrast's name, as a result records it: the file's stem."""
    return Path(name_or_path).stem


def role_context(role: Role, q: str, r: str | None = None) -> str:
    """The role's context up to its answer: one user turn with Q, or Q, a
    blank line and R, then the empty think block. With ``reads_r ==
    "think"``, R fills the role's think block instead."""
    if r == "":
        raise ValueError("R is empty")
    r_as_text = r is not None and role.reads_r == "text"
    text = f"{q}\n\n{r}" if r_as_text else q
    user_turn = QWEN35.user_turn(role.template.replace(PLACEHOLDER, text))
    if r is None or r_as_text:
        return user_turn + QWEN35.EMPTY_THINK
    return user_turn + QWEN35.THINK_OPEN + QWEN35.closed_reasoning(r)
