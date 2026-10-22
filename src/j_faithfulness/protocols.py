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


# Protocol 1 gives the model a role before Q: the decide task of the
# contrastive read-out, so J (read out under it) comes from the context it's
# injected into.
ROLE = "You are an advisor. Decide the following question."


def lead(role: str | None) -> str:
    return f"{role}\n\n" if role else ""


def decision_prompt(q: str, role: str | None = ROLE) -> str:
    """Protocol 1 up to D: the role and Q, then the assistant turn with
    thinking off. Its last token is where D is produced."""
    return user_turn(lead(role) + q) + DECISION_OPEN


def decision_skip(role: str | None = ROLE) -> str:
    """The decision prompt's leading text up to Q, which read-outs skip."""
    return USER_OPEN + lead(role)


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
    "decide": ROLE,
}


def task_prompt(q: str, task: str) -> tuple[str, str]:
    """Protocol 1 up to D with a task from READOUT_TASKS before Q. Returns the
    prompt and its leading text up to Q, which read-outs skip."""
    role = READOUT_TASKS[task]
    return decision_prompt(q, role), decision_skip(role)


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
