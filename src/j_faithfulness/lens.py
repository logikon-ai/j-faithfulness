"""Load the model and its pre-fitted lens; read out a case word by word under
the advisor and the contrast roles, aligned."""

from __future__ import annotations

import functools
from bisect import bisect_right
from dataclasses import dataclass
from typing import NamedTuple

import jlens
import torch
from jlens.hooks import ActivationRecorder

from j_faithfulness.contrasts import Role, role_context
from j_faithfulness.formats import QWEN35, Kind

LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"


@dataclass(frozen=True)
class LensSpec:
    """A model's pre-fitted lens file and the band of layers J is read from
    (DESIGN.md, Latent readout)."""

    file: str
    band: tuple[int, int]


LENSES: dict[str, LensSpec] = {
    # The known intermediates rank near the top from layers 6–9, and at the
    # token before D the answer enters the top 10 at layer 22.
    "Qwen/Qwen3.5-4B": LensSpec(
        "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        (8, 21),
    ),
    # Placeholder: the 4B's band scaled from 32 to 64 layers, not yet retuned.
    "Qwen/Qwen3.5-27B": LensSpec(
        "qwen3.5-27b/jlens/Salesforce-wikitext/Qwen3.5-27B_jacobian_lens.pt",
        (16, 42),
    ),
}

# lens.apply truncates from the right at max_seq_len, which would cut off the
# end of the context. Read-outs pass this and fail if the context reaches it.
MAX_SEQ_LEN = 32768

# Positions unembedded at once: the logits over a 250k vocabulary are the
# large tensor, not the forward pass. 2048 rows of float32 log-probabilities
# take 2 GB, a few copies of which fit easily on molab's 96 GB; fewer, larger
# calls keep the GPU busy.
CHUNK = 2048

# Units whose peaks are held at once: the [units, vocab] float32 buffer takes
# 1 MB per unit, so a long context with tens of thousands of word units would
# not fit in one buffer.
UNIT_BLOCK = 2048


@functools.cache
def load(model_id: str, revision: str | None = None):
    """Load the model (bf16, CUDA) at ``revision`` and its lens from
    :data:`LENSES`, once per id and revision.

    Returns ``(hf_model, model, lens)``: the HF model, which scoring needs
    for its attention mask, its jlens wrapper, and the
    :class:`jlens.JacobianLens`.
    """
    import transformers

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = jlens.from_hf(hf_model, tokenizer)
    spec = LENSES[model_id]
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=spec.file, revision=LENS_REVISION
    )
    # Fail before a long run, not inside it: a lens fitted on fewer layers
    # than the band would raise only at the first read-out.
    missing = set(range(spec.band[0], spec.band[1] + 1)) - set(lens.source_layers)
    if missing:
        raise ValueError(f"{model_id}: lens has no layers {sorted(missing)}")
    return hf_model, model, lens


# Raw tokens kept per unit and layer: enough for word-like ones to remain
# once the rest is filtered out.
TOP_N = 200


@dataclass(frozen=True)
class LatentReadout:
    """Lens read-out over the units of a context (DESIGN.md, Units).

    ``top[u][j]`` are the decoded top tokens of unit ``u`` at the band's
    ``j``-th layer, in rank order (index = rank), unfiltered;
    :func:`j_faithfulness.verbalize.j_and_reversed` turns them into J. The first
    ``q_units`` units are Q's, which IDF is taken over.
    """

    context: str
    top: list[list[list[str]]]
    q_units: int


def word_ends(tokens: list[str]) -> list[int]:
    """Positions that end a word: every token except a subtoken followed by
    the rest of its word. Punctuation and special tokens are kept."""
    ends = []
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        continues = token[-1:].isalnum() and nxt[:1].isalnum()
        if not continues:
            ends.append(i)
    return ends


def sentence_ids(tokens: list[str]) -> list[int]:
    """The sentence each token belongs to; a sentence ends with a token that
    holds ``.``, ``?``, ``!`` or a line break (DESIGN.md, Units). Whitespace
    after the end stays with the sentence, so a bare line break isn't a
    sentence of its own."""
    ids, sentence, ended = [], 0, False
    for token in tokens:
        if ended and token.strip():
            sentence += 1
            ended = False
        ids.append(sentence)
        if any(mark in token for mark in ".?!\n"):
            ended = True
    return ids


def tokenize(tokenizer, context: str) -> tuple[list[str], list[tuple[int, int]]]:
    """The tokens of ``context``, decoded one by one, and their character
    offsets. ``tokenizer`` must give offsets, and its tokens must be the
    model's."""
    encoding = tokenizer(context, return_offsets_mapping=True)
    tokens = [
        tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in encoding["input_ids"]
    ]
    return tokens, encoding["offset_mapping"]


def units(
    tokens: list[str],
    offsets: list[tuple[int, int]],
    start: int,
    sentences: range | None = None,
) -> list[list[int]]:
    """The read-out units of a tokenized context from character ``start`` on
    (DESIGN.md, Units): each word is one unit at its last token, and within
    the characters ``sentences`` (R) each sentence is one unit of all its
    tokens."""

    def in_sentences(p: int) -> bool:
        return sentences is not None and offsets[p][0] in sentences

    out = [
        [p] for p in word_ends(tokens) if offsets[p][0] >= start and not in_sentences(p)
    ]
    inside = [p for p in range(len(tokens)) if in_sentences(p)]
    grouped: dict[int, list[int]] = {}
    for p, sentence in zip(
        inside, sentence_ids([tokens[p] for p in inside]), strict=True
    ):
        grouped.setdefault(sentence, []).append(p)
    return sorted(out + list(grouped.values()))


def top_per_unit(
    states: torch.Tensor,
    read_units: list[list[int]],
    unembed,
    top_n: int,
) -> list[list[int]]:
    """The ``top_n`` token ids of each unit, ranked by their highest
    log-probability over the unit's positions in ``states`` ([seq, d_model]),
    which ``unembed`` maps to logits.

    All positions are unembedded together, in chunks of CHUNK rows, and a
    segment-max reduces the rows to one per unit. Each row is a (position,
    unit) pair, so a position shared by several units counts for each; units
    are disjoint in practice, and an overlap costs only a repeated unembedding.
    """
    if any(not unit for unit in read_units):
        raise ValueError("every unit needs at least one position")
    ranked: list[list[int]] = []
    for b in range(0, len(read_units), UNIT_BLOCK):
        block = read_units[b : b + UNIT_BLOCK]
        rows = [p for unit in block for p in unit]
        unit_of_row = torch.tensor([u for u, unit in enumerate(block) for _ in unit])
        peak = None
        for c in range(0, len(rows), CHUNK):
            log_probs = unembed(states[rows[c : c + CHUNK]]).float().log_softmax(-1)
            if peak is None:
                peak = log_probs.new_full((len(block), log_probs.shape[-1]), -torch.inf)
                unit_of_row = unit_of_row.to(log_probs.device)
            # The max over a unit doesn't depend on how its rows are chunked.
            index = unit_of_row[c : c + CHUNK, None].expand_as(log_probs)
            peak.scatter_reduce_(0, index, log_probs, reduce="amax")
        ranked += peak.topk(min(top_n, peak.shape[-1])).indices.tolist()
    return ranked


@torch.no_grad()
def read_out(
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    context: str,
    band: tuple[int, int],
    units: list[list[int]],
    q_units: int,
) -> LatentReadout:
    """Read out the lens at every layer of ``band`` (inclusive), unit by unit
    (see :func:`units`): at each layer, a token scores its highest
    log-probability over the unit's positions. Max, not mean: an
    intermediate often peaks at one word and would be averaged away over a
    sentence. The first ``q_units`` units are Q's."""
    layers = list(range(band[0], band[1] + 1))
    input_ids = model.encode(context, max_length=MAX_SEQ_LEN)
    if input_ids.shape[-1] >= MAX_SEQ_LEN:
        raise ValueError(f"context reaches MAX_SEQ_LEN={MAX_SEQ_LEN} tokens")

    vocab: dict[int, str] = {}

    def decode(i: int) -> str:
        if i not in vocab:
            vocab[i] = model.tokenizer.decode([i], clean_up_tokenization_spaces=False)
        return vocab[i]

    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {layer: recorder.activations[layer][0] for layer in layers}

    top: list[list[list[str]]] = [[] for _ in units]
    for layer in layers:
        # One transport per layer: the Jacobian moves to the device once.
        transported = lens.transport(activations.pop(layer).float(), layer)
        ranked = top_per_unit(transported, units, model.unembed, TOP_N)
        for u, ids in enumerate(ranked):
            top[u].append([decode(i) for i in ids])
    return LatentReadout(context, top, q_units)


class Segments(NamedTuple):
    """The units of a context from Q on, by the text they sit in. Without R,
    every unit after Q is in ``after``."""

    q: list[list[int]]
    between: list[list[int]]
    r: list[list[int]]
    after: list[list[int]]


def segments(tokenizer, context: str, q: str, r: str | None) -> Segments:
    """The units of ``context`` from Q on (see :func:`units`), split at Q's
    and R's ends by where each unit starts."""
    q_start = context.index(q)
    q_end = q_start + len(q)
    tokens, offsets = tokenize(tokenizer, context)
    if r is None:
        read_units = units(tokens, offsets, q_start)
        q_units = [u for u in read_units if offsets[u[0]][0] < q_end]
        return Segments(q_units, [], [], read_units[len(q_units) :])
    r_start = context.index(r, q_end)
    r_end = r_start + len(r)
    split = Segments([], [], [], [])
    for unit in units(tokens, offsets, q_start, range(r_start, r_end)):
        split[bisect_right((q_end, r_start, r_end), offsets[unit[0]][0])].append(unit)
    return split


def align(advisor: Segments, role: Segments) -> list[list[int]]:
    """The role's units in the advisor's order (DESIGN.md, Latent readout),
    so the readouts are graded unit by unit. Q's and R's units pair one to
    one. An advisor unit between Q and R gets the role's last unit before R,
    where both have read Q and nothing of R. An advisor unit after the last
    of Q and R gets the role's last unit, where both have read everything."""
    if len(advisor.q) != len(role.q) or len(advisor.r) != len(role.r):
        raise ValueError("Q or R splits into different units for the two roles")
    before_r = role.q + role.between
    last = (before_r + role.r + role.after)[-1]
    return (
        role.q
        + [before_r[-1]] * len(advisor.between)
        + role.r
        + [last] * len(advisor.after)
    )


def read_case(
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    band: tuple[int, int],
    roles: list[Role],
    q: str,
    kind: Kind,
    r: str | None = None,
) -> tuple[LatentReadout, list[LatentReadout]]:
    """Read out J's span for one case (DESIGN.md, Latent readout, Span)
    under the advisor and under each role, the roles' units aligned to the
    advisor's (see :func:`align`). The span runs from Q up to R in the
    reasoning case, or up to D in the explanation case, R included if there
    is one. ``r``: the transcript's R, None with thinking off; only the
    explanation case reads it. ``band``: the model's, from :data:`LENSES`.

    Returns the advisor's readout and the roles' readouts, in ``roles``'
    order.
    """
    if kind == "reasoning":
        advisor_context, read_r = QWEN35.reasoning_prompt(q), None
    else:
        advisor_context, read_r = QWEN35.decision_prompt(q, r), r
    advisor_segments = segments(model.tokenizer, advisor_context, q, read_r)
    q_units = len(advisor_segments.q)
    advisor_units = [unit for part in advisor_segments for unit in part]
    advisor = read_out(model, lens, advisor_context, band, advisor_units, q_units)
    contrast = []
    for role in roles:
        context = role_context(role, q, read_r)
        role_units = align(
            advisor_segments, segments(model.tokenizer, context, q, read_r)
        )
        contrast.append(read_out(model, lens, context, band, role_units, q_units))
    return advisor, contrast
