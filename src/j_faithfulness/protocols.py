"""Raw-string transcripts (see DESIGN.md, Conversation protocols)."""

from __future__ import annotations

import torch

EMPTY_THINK = "<think>\n\n</think>\n\n"
# Opens every transcript. Its read-out is the same template junk on every
# prompt, so read-outs skip it.
USER_OPEN = "<|im_start|>user\n"
# The assistant turn with thinking off, up to D.
DECISION_OPEN = "<|im_start|>assistant\n" + EMPTY_THINK


def user_turn(text: str) -> str:
    return f"{USER_OPEN}{text}<|im_end|>\n"


def decision_prompt(q: str) -> str:
    """Protocol 1 up to D: Q, then the assistant turn with thinking off. Its last
    token is where D is produced."""
    return user_turn(q) + DECISION_OPEN


# Task prefixes before Q for the contrastive read-out (PLAN.md, M1). Each gives
# the model a role; the baselines tell it not to answer, so it has no reason to
# think the problem through. They go before Q because a task after Q can't
# change the read-out at Q's positions.
READOUT_TASKS = {
    "copy": (
        "You are a typist. Retype the following text exactly as it is. Do not "
        "answer or comment on any question in it."
    ),
    "plain": (
        "You are an editor. Rewrite the following text in easy-to-read, plain "
        "English, keeping its meaning. Do not answer any question in it."
    ),
    "decide": "You are an advisor. Decide the following question.",
}


def task_prompt(q: str, task: str) -> tuple[str, str]:
    """Protocol 1 up to D with a task from READOUT_TASKS before Q. Returns the
    prompt and its leading text up to Q, which read-outs skip."""
    skip = f"{USER_OPEN}{READOUT_TASKS[task]}\n\n"
    return decision_prompt(f"{READOUT_TASKS[task]}\n\n{q}"), skip


def option_distribution(
    logits: torch.Tensor, option_ids: list[int], options: list[str]
) -> dict[str, float]:
    """Next-token probabilities of the option tokens, from logits over the
    vocabulary. Not renormalized: the mass left over went to other tokens."""
    p = torch.softmax(logits.float(), -1)
    return {o: p[i].item() for o, i in zip(options, option_ids, strict=True)}


@torch.no_grad()
def option_probs(
    hf_model, tokenizer, prompt: str, options: list[str]
) -> dict[str, float]:
    """P(D = o) for each single-token option ``o`` after ``prompt``."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(hf_model.device)
    logits = hf_model(ids).logits[0, -1]
    return option_distribution(
        logits, tokenizer.convert_tokens_to_ids(options), options
    )


def decision(probs: dict[str, float]) -> str:
    """D: the most probable option."""
    return max(probs, key=probs.get)
