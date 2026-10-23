import math
from types import SimpleNamespace

import pytest
import torch

from j_faithfulness.protocols import (
    CONCISE,
    ROLE,
    Trace,
    batch_option_probs,
    decision,
    decision_prompt,
    decision_skip,
    mean_probs,
    option_distribution,
    option_probs,
    sample_reasoning,
    task_prompt,
    trace,
)


def test_decision_prompt_matches_template():
    # apply_chat_template([user Q], add_generation_prompt=True,
    # enable_thinking=False) for Qwen3.5-4B.
    assert decision_prompt("Q?", None) == (
        "<|im_start|>user\nQ?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert decision_prompt("Q?") == decision_prompt(f"{ROLE}\n\nQ?", None)
    assert decision_prompt("Q?").startswith(decision_skip())


def test_task_prompt():
    prompt, skip = task_prompt("Q?", "decide")
    assert skip == (
        "<|im_start|>user\nYou are an advisor. Decide the following question.\n\n"
    )
    assert (
        prompt == skip + "Q?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert (prompt, skip) == (decision_prompt("Q?"), decision_skip())


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


def test_thinking_prompts():
    role = (
        "You are an advisor. Decide the following question. Keep your internal "
        "reasoning concise."
    )
    assert decision_prompt("Q?", thinking=True) == (
        f"<|im_start|>user\n{role}\n\nQ?<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    assert decision_skip(thinking=True) == f"<|im_start|>user\n{role}\n\n"
    # The decide task with thinking on is the protocol-2 prompt; the copy task
    # gets the same suffix.
    assert task_prompt("Q?", "decide", True) == (
        decision_prompt("Q?", thinking=True),
        decision_skip(thinking=True),
    )
    assert task_prompt("Q?", "copy", True)[1].endswith(" " + CONCISE + "\n\n")


def test_trace():
    p = decision_prompt("Q?", thinking=True)
    t = trace(p, "Weigh it.\n</think>")
    assert t == Trace(p, "Weigh it.", False)
    # The transcript up to D matches the template's format for a closed R.
    assert t.transcript == p + "Weigh it.\n</think>\n\n"
    # Without </think>, R hit the cap and is closed the same way.
    t = trace(p, "Weigh")
    assert t.truncated and t.transcript == p + "Weigh\n</think>\n\n"


class _Tokenizer:
    """Token ids: 0 = pad, 1 = </think>, 2 = <|im_end|>, 10+ = letters;
    every prompt is one token."""

    pad_token_id = 0

    def __call__(self, texts, return_tensors, padding, padding_side):
        assert padding_side == "left"
        return SimpleNamespace(
            input_ids=torch.full((len(texts), 1), 99),
            attention_mask=torch.ones(len(texts), 1, dtype=torch.long),
        )

    def convert_tokens_to_ids(self, tokens):
        names = {"</think>": 1, "A": 10, "B": 11}
        return names[tokens] if isinstance(tokens, str) else [names[t] for t in tokens]

    def decode(self, ids, skip_special_tokens):
        names = {0: "<pad>", 1: "</think>", 2: "<|im_end|>"}
        return "".join(names.get(i, chr(ord("a") + i - 10)) for i in ids)


def test_sample_reasoning_cuts_at_think_end():
    # Two prompts, two samples each; generate returns them prompt by prompt.
    rows = [
        [99, 10, 11, 1, 0, 0],  # closed
        [99, 10, 11, 12, 13, 14],  # hit the cap
        [99, 10, 2, 0, 0, 0],  # ended the turn without </think>
        [99, 12, 1, 0, 0, 0],  # closed
    ]

    class Model:
        device = torch.device("cpu")
        generation_config = SimpleNamespace(eos_token_id=[2])

        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs["eos_token_id"] == [1, 2]
            assert kwargs["num_return_sequences"] == 2
            return torch.tensor(rows)

    traces = sample_reasoning(Model(), _Tokenizer(), ["P", "Q"], n=2, max_tokens=5)
    assert [[(t.prompt, t.r, t.truncated) for t in ts] for ts in traces] == [
        [("P", "ab", False), ("P", "abcde", True)],
        [("Q", "a", True), ("Q", "c", False)],
    ]


def test_batch_option_probs():
    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, attention_mask, logits_to_keep):
            # Row r puts probability 0.1 * (r + 1) on A, 0.5 on B.
            logits = torch.full((input_ids.shape[0], 1, 12), -1e9)
            for r in range(input_ids.shape[0]):
                logits[r, -1, 10] = math.log(0.1 * (r + 1))
                logits[r, -1, 11] = math.log(0.5)
                logits[r, -1, 0] = math.log(0.5 - 0.1 * (r + 1))
            return SimpleNamespace(logits=logits)

    probs = batch_option_probs(Model(), _Tokenizer(), ["P"] * 3, ["A", "B"], 2)
    # Batches of 2, then 1: rows restart at 0 in each batch.
    assert [p["A"] for p in probs] == pytest.approx([0.1, 0.2, 0.1])
    assert [p["B"] for p in probs] == pytest.approx([0.5] * 3)


def test_mean_probs():
    assert mean_probs([{"A": 0.8, "B": 0.2}, {"A": 0.4, "B": 0.4}]) == pytest.approx(
        {"A": 0.6, "B": 0.3}
    )
