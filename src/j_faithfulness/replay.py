"""Record GPU results on molab and replay them locally. Temporary: it lets
the demo notebook's cells be tuned without a GPU, and goes before the first
release.

Two environment variables, read at each call:
- ``J_FAITHFULNESS_RECORD=path``: run for real and add each result to path.
- ``J_FAITHFULNESS_REPLAY=path``: return the results stored in path; no
  weights are loaded.
"""

from __future__ import annotations

import dataclasses
import functools
import gzip
import hashlib
import inspect
import json
import os
from types import SimpleNamespace


def record_path() -> str | None:
    return os.environ.get("J_FAITHFULNESS_RECORD")


def replay_path() -> str | None:
    return os.environ.get("J_FAITHFULNESS_REPLAY")


def read(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write(path: str, store: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(store, f)


def stand_in(model_id: str):
    """What ``load`` returns when replaying: a model that only knows its layer
    count (from the config, no weights) and its tokenizer, and no lens."""
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id)
    model = SimpleNamespace(
        n_layers=config.get_text_config().num_hidden_layers,
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_id),
    )
    return model, None


def recorded(result_type):
    """Record or replay a function of ``(model, lens, ...)`` that returns a
    ``result_type`` dataclass, keyed by its name and its other arguments."""

    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            replaying, recording = replay_path(), record_path()
            if not (replaying or recording):
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            # Arguments left at None stay out of the key, so a new optional
            # argument keeps the earlier recordings valid.
            key_args = {
                k: v
                for k, v in bound.arguments.items()
                if k not in ("model", "lens") and v is not None
            }
            digest = hashlib.sha256(json.dumps(key_args, sort_keys=True).encode())
            key = f"{fn.__name__}:{digest.hexdigest()[:16]}"
            if replaying:
                store = read(replaying)
                if key not in store:
                    raise KeyError(f"{key} isn't in {replaying}: record it on molab")
                return result_type(**store[key])
            result = fn(*args, **kwargs)
            store = read(recording)
            store[key] = dataclasses.asdict(result)
            write(recording, store)
            return result

        return wrapper

    return decorate
