from types import SimpleNamespace

import pytest
import torch

from j_faithfulness.inject import NOTICE, injected_decision_prompt
from j_faithfulness.protocols import (
    decision,
    decision_prompt,
    option_distribution,
    option_probs,
)


def test_decision_prompt_matches_template():
    # apply_chat_template([user Q], add_generation_prompt=True,
    # enable_thinking=False) for Qwen3.5-4B.
    assert decision_prompt("Q?") == (
        "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_injected_decision_prompt():
    tail = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert injected_decision_prompt("Q?", "<latentthought>\na.\n</latentthought>") == (
        f"<|im_start|>user\n{NOTICE}\n\nQ?\n\n<latentthought>\na.\n</latentthought>"
        + tail
    )
    assert injected_decision_prompt("Q?") == f"<|im_start|>user\n{NOTICE}\n\nQ?" + tail
    assert injected_decision_prompt("Q?", notice=None) == decision_prompt("Q?")


def test_option_distribution_is_not_renormalized():
    logits = torch.log(torch.tensor([0.5, 0.2, 0.3]))
    p = option_distribution(logits, [0, 2], ["A", "B"])
    assert p == pytest.approx({"A": 0.5, "B": 0.3})
    assert decision(p) == "A"


def test_option_probs_reads_the_last_position():
    class Model:
        device = torch.device("cpu")

        def __call__(self, ids):
            logits = torch.zeros(1, ids.shape[-1], 3)
            logits[0, -1] = torch.log(torch.tensor([0.1, 0.6, 0.3]))
            return SimpleNamespace(logits=logits)

    class Tokenizer:
        def __call__(self, text, return_tensors):
            return SimpleNamespace(
                input_ids=torch.zeros(1, len(text), dtype=torch.long)
            )

        def convert_tokens_to_ids(self, tokens):
            return [{"A": 1, "B": 2}[t] for t in tokens]

    p = option_probs(Model(), Tokenizer(), "Q?", ["A", "B"])
    assert p == pytest.approx({"A": 0.6, "B": 0.3})
