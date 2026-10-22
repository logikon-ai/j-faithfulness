"""Raw-string transcripts (see DESIGN.md, Conversation protocols)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

EMPTY_THINK = "<think>\n\n</think>\n\n"
# Opens every transcript. Its read-out is the same template junk on every
# prompt, so read-outs skip it.
USER_OPEN = "<|im_start|>user\n"
# The assistant turn with thinking off, up to D.
DECISION_OPEN = "<|im_start|>assistant\n" + EMPTY_THINK
# The assistant turn with thinking on, up to R.
THINK_OPEN = "<|im_start|>assistant\n<think>\n"
THINK_END = "</think>"


def opening(thinking: bool) -> str:
    return THINK_OPEN if thinking else DECISION_OPEN


def user_turn(text: str) -> str:
    return f"{USER_OPEN}{text}<|im_end|>\n"


# Protocol 1 gives the model a role before Q: the decide task of the
# contrastive read-out, so J (read out under it) comes from the context it's
# injected into.
ROLE = "You are an advisor. Decide the following question."
# With thinking on, the role line also steers R's length (DESIGN.md, Comparing
# outcomes); R is sampled with the model's default settings.
CONCISE = "Keep your internal reasoning concise."


def lead(role: str | None, thinking: bool = False) -> str:
    """The role line before Q; with thinking on, it ends in CONCISE."""
    if not role:
        return ""
    return f"{role} {CONCISE}\n\n" if thinking else f"{role}\n\n"


def decision_prompt(q: str, role: str | None = ROLE, thinking: bool = False) -> str:
    """Protocol 1 up to D: the role and Q, then the assistant turn with
    thinking off. Its last token is where D is produced. ``thinking=True``
    gives protocol 2 up to R instead, ending in ``<think>\n``."""
    return user_turn(lead(role, thinking) + q) + opening(thinking)


def decision_skip(role: str | None = ROLE, thinking: bool = False) -> str:
    """The decision prompt's leading text up to Q, which read-outs skip."""
    return USER_OPEN + lead(role, thinking)


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


def task_prompt(q: str, task: str, thinking: bool = False) -> tuple[str, str]:
    """Protocol 1 up to D with a task from READOUT_TASKS before Q. Returns the
    prompt and its leading text up to Q, which read-outs skip.
    ``thinking=True``: protocol 2 up to R."""
    role = READOUT_TASKS[task]
    return decision_prompt(q, role, thinking), decision_skip(role, thinking)


@dataclass(frozen=True)
class Trace:
    """A sampled R after ``prompt`` (which ends in ``<think>\n``).
    ``truncated``: R didn't reach ``</think>`` (it hit the token cap) and was
    closed by us."""

    prompt: str
    r: str
    truncated: bool

    @property
    def transcript(self) -> str:
        """Protocol 2 up to D: the prompt, R, then ``</think>``; D is produced
        at its last token."""
        return f"{self.prompt}{self.r}\n{THINK_END}\n\n"


def trace(prompt: str, generated: str) -> Trace:
    """The Trace from the text generated after ``prompt``: R up to
    ``</think>``, without the newline the model puts before it. Without
    ``</think>``, the generation hit the cap."""
    r, closed, _ = generated.partition(THINK_END)
    if not closed:
        return Trace(prompt, generated, True)
    return Trace(prompt, r.removesuffix("\n"), False)


def left_padded(tokenizer, prompts: list[str], device) -> dict:
    """``prompts`` tokenized as one batch, padded on the left so every row
    ends at its last token."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    return {
        "input_ids": enc.input_ids.to(device),
        "attention_mask": enc.attention_mask.to(device),
    }


@torch.no_grad()
def sample_reasoning(
    hf_model, tokenizer, prompts: list[str], n: int = 8, max_tokens: int = 4000
) -> list[list[Trace]]:
    """``n`` R's after each of ``prompts``, all sampled in one batch with the
    model's default settings, each stopped at ``</think>`` or after
    ``max_tokens``."""
    enc = left_padded(tokenizer, prompts, hf_model.device)
    stop = tokenizer.convert_tokens_to_ids(THINK_END)
    eos = hf_model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    out = hf_model.generate(
        **enc,
        do_sample=True,
        max_new_tokens=max_tokens,
        num_return_sequences=n,
        eos_token_id=[stop, *eos],
        pad_token_id=tokenizer.pad_token_id,
    )
    # generate returns the n samples of each prompt next to each other.
    rows = out[:, enc["input_ids"].shape[-1] :].tolist()
    traces = []
    for k, prompt in enumerate(prompts):
        traces.append([])
        for row in rows[k * n : (k + 1) * n]:
            # Keep R up to and including </think>; an end-of-turn token before
            # it (and the padding after) is dropped, and R is closed as if
            # truncated.
            end = next((i for i, t in enumerate(row) if t in [stop, *eos]), len(row))
            row = row[: end + 1] if end < len(row) and row[end] == stop else row[:end]
            text = tokenizer.decode(row, skip_special_tokens=False)
            traces[-1].append(trace(prompt, text))
    return traces


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


@torch.no_grad()
def batch_option_probs(
    hf_model, tokenizer, prompts: list[str], options: list[str], batch_size: int = 8
) -> list[dict[str, float]]:
    """``option_probs`` for each of ``prompts``, ``batch_size`` at a time;
    only the last position's logits are computed."""
    ids = tokenizer.convert_tokens_to_ids(options)
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = left_padded(tokenizer, prompts[i : i + batch_size], hf_model.device)
        logits = hf_model(**enc, logits_to_keep=1).logits[:, -1]
        out += [option_distribution(row, ids, options) for row in logits]
    return out


def mean_probs(dists: list[dict[str, float]]) -> dict[str, float]:
    """The answer distribution averaged over sampled R's."""
    return {o: sum(d[o] for d in dists) / len(dists) for o in dists[0]}


def decision(probs: dict[str, float]) -> str:
    """D: the most probable option."""
    return max(probs, key=probs.get)
