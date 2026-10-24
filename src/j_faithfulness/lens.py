"""Load the model and its pre-fitted lens; read the lens out word by word."""

from __future__ import annotations

import functools
from bisect import bisect_right
from dataclasses import dataclass
from typing import NamedTuple

import jlens
import torch
from jlens.hooks import ActivationRecorder

from j_faithfulness.formats import Kind, Qwen35Format

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
def load(model_id: str):
    """Load the model (bf16, CUDA) and its lens from :data:`LENSES`, once per
    id.

    Returns ``(model, lens)``: the jlens wrapper of the HF model, and the
    :class:`jlens.JacobianLens`.
    """
    import transformers

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
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
    return model, lens


# Raw tokens kept per unit and layer: enough for word-like ones to remain
# once the rest is filtered out.
TOP_N = 200


@dataclass(frozen=True)
class LatentReadout:
    """Lens read-out over the units of a context (DESIGN.md, Units).

    ``top[u][j]`` are the decoded top tokens of unit ``u`` at the band's
    ``j``-th layer, in rank order (index = rank), unfiltered;
    :func:`j_faithfulness.verbalize.verbalize` turns them into J. The first
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
    """The units of a context with Q and R, by the text they sit in."""

    q: list[list[int]]
    before_r: list[list[int]]
    r: list[list[int]]
    after_r: list[list[int]]


def segments(tokenizer, context: str, q: str, r: str) -> Segments:
    """The units of ``context`` from Q on (see :func:`units`), split at Q's
    and R's ends by where each unit starts."""
    q_start = context.index(q)
    q_end = q_start + len(q)
    r_start = context.index(r, q_end)
    r_end = r_start + len(r)
    tokens, offsets = tokenize(tokenizer, context)
    split = Segments([], [], [], [])
    for unit in units(tokens, offsets, q_start, range(r_start, r_end)):
        split[bisect_right((q_end, r_start, r_end), offsets[unit[0]][0])].append(unit)
    return split


def align_to_advisor(advisor: Segments, editor: Segments) -> list[list[int]]:
    """The editor's units in the advisor's order (DESIGN.md, Two roles), so
    the two read-outs are graded unit by unit. Q's and R's units pair one to
    one. Each advisor unit between Q and R gets the editor's last unit before
    R: both have read Q and nothing of R. The advisor's units after R get the
    editor's last ones, which close the same template."""
    if len(advisor.q) != len(editor.q) or len(advisor.r) != len(editor.r):
        raise ValueError("Q or R splits into different units for the two roles")
    if not editor.before_r or len(editor.after_r) < len(advisor.after_r):
        raise ValueError("the editor's context has too few units around R")
    after_r = editor.after_r[len(editor.after_r) - len(advisor.after_r) :]
    return editor.q + [editor.before_r[-1]] * len(advisor.before_r) + editor.r + after_r


def read_span(
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    fmt: Qwen35Format,
    kind: Kind,
    q: str,
    r: str | None = None,
    *,
    editor: bool = False,
    band: tuple[int, int],
) -> LatentReadout:
    """Read out J's span for ``kind`` (DESIGN.md, Latent readout, Span), unit
    by unit, from Q on: Q and the generation prompt up to R, or up to D,
    with R's sentences as units if there is an R. ``editor``: read the copy
    editor's context instead of the advisor's, with its units aligned to
    the advisor's. ``band``: the model's, from :data:`LENSES`."""
    if kind == "reasoning":
        context = fmt.reasoning_prompt(q, fmt.EDIT_ROLE if editor else None)
    elif editor:
        context = fmt.editor_prompt(q, r)
    else:
        context = fmt.decision_prompt(q, r)
    if kind == "explanation" and r is not None and editor:
        own = segments(model.tokenizer, context, q, r)
        span_units = align_to_advisor(
            segments(model.tokenizer, fmt.decision_prompt(q, r), q, r), own
        )
        q_units = len(own.q)
    else:
        start = context.index(q)
        q_end = start + len(q)
        sentences = None
        if kind == "explanation" and r is not None:
            r_start = context.index(r, q_end)
            sentences = range(r_start, r_start + len(r))
        tokens, offsets = tokenize(model.tokenizer, context)
        span_units = units(tokens, offsets, start, sentences)
        q_units = sum(offsets[unit[0]][0] < q_end for unit in span_units)
    return read_out(model, lens, context, band, span_units, q_units)
