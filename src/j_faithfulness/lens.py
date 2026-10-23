"""Load the model and its pre-fitted lens; read the lens out word by word."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jlens
import torch
from jlens.hooks import ActivationRecorder

from j_faithfulness.formats import Kind, Span, TranscriptFormat

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
LENS_REVISION = "qwen-n1000"

# Mid-network layers J is read from (DESIGN.md, Latent readout): the known
# intermediates rank near the top from layers 6–9, and at the token before D
# the answer enters the top 10 at layer 22.
BAND = (8, 21)

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
def load(model_id: str = MODEL_NAME):
    """Load Qwen3.5-4B (bf16, CUDA) and its lens, once per id.

    Returns ``(model, lens)``: the jlens wrapper of the HF model, and the
    :class:`jlens.JacobianLens`.
    """
    import transformers

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    return model, lens


@dataclass
class LatentReadout:
    """Lens read-out over a span of positions.

    ``top[i][j]`` are the decoded top tokens at ``positions[i]`` and
    ``layers[j]``, in rank order (index = rank), unfiltered;
    :func:`j_faithfulness.verbalize.verbalize` turns them into J.
    ``context_tokens`` is the whole context, decoded token by token.

    Each position stands for a unit (DESIGN.md, Units): ``units[i]`` holds
    its positions, of which ``positions[i]`` is the last; None means every
    unit is its one position. The first ``q_units`` units are Q's, which IDF
    is taken over; None means all.
    """

    context: str
    context_tokens: list[str]
    positions: list[int]
    layers: list[int]
    top: list[list[list[str]]]
    units: list[list[int]] | None = None
    q_units: int | None = None


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


def skipped_tokens(tokens: list[str], skip: str) -> int:
    """The number of leading tokens that spell out ``skip`` exactly (0 if they
    don't)."""
    text = ""
    for i, token in enumerate(tokens):
        if text == skip:
            return i
        text += token
        if not skip.startswith(text):
            return 0
    return len(tokens) if text == skip else 0


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


def units(
    tokenizer,
    context: str,
    start: int,
    sentences: Span | None = None,
    end: int | None = None,
) -> list[list[int]]:
    """The read-out units of ``context`` from character ``start`` on
    (DESIGN.md, Units): each word is one unit at its last token, and within
    ``sentences`` (R) each sentence is one unit of all its tokens. With
    ``end``, only the units that start before it, e.g. Q's.

    ``tokenizer`` must give offsets, and its tokens must be the model's.
    """
    encoding = tokenizer(context, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    tokens = [
        tokenizer.decode([i], clean_up_tokenization_spaces=False)
        for i in encoding["input_ids"]
    ]

    def in_sentences(p: int) -> bool:
        return (
            sentences is not None and sentences.start <= offsets[p][0] < sentences.end
        )

    out = [
        [p] for p in word_ends(tokens) if offsets[p][0] >= start and not in_sentences(p)
    ]
    inside = [p for p in range(len(tokens)) if in_sentences(p)]
    grouped: dict[int, list[int]] = {}
    for p, sentence in zip(
        inside, sentence_ids([tokens[p] for p in inside]), strict=True
    ):
        grouped.setdefault(sentence, []).append(p)
    out = sorted(out + list(grouped.values()))
    if end is not None:
        out = [unit for unit in out if offsets[unit[0]][0] < end]
    return out


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
    positions: list[int] | None = None,
    skip: str = "",
    top_n: int = 200,
    units: list[list[int]] | None = None,
) -> LatentReadout:
    """Read out the lens at every layer of ``band`` (inclusive) and at
    ``positions`` (default: every word end of ``context`` after the leading
    text ``skip``, e.g. the opening template of the user turn).

    With ``units`` (see :func:`units`), each unit is read out as one: at each
    layer, a token scores its highest log-probability over the unit's
    positions. Max, not mean: an intermediate often peaks at one word and
    would be averaged away over a sentence.

    ``top_n`` raw tokens are kept per unit and layer: enough for word-like
    ones to remain once the rest is filtered out.
    """
    layers = list(range(band[0], band[1] + 1))
    input_ids = model.encode(context, max_length=MAX_SEQ_LEN)
    n = input_ids.shape[-1]
    if n >= MAX_SEQ_LEN:
        raise ValueError(f"context reaches MAX_SEQ_LEN={MAX_SEQ_LEN} tokens")

    vocab: dict[int, str] = {}

    def decode(i: int) -> str:
        if i not in vocab:
            vocab[i] = model.tokenizer.decode([i], clean_up_tokenization_spaces=False)
        return vocab[i]

    context_tokens = [decode(i) for i in input_ids[0].tolist()]
    if units is None:
        if positions is None:
            start = skipped_tokens(context_tokens, skip)
            positions = [p for p in word_ends(context_tokens) if p >= start]
        read_units = [[p % n] for p in positions]
    else:
        read_units = [[p % n for p in unit] for unit in units]

    with ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(input_ids)
        activations = {layer: recorder.activations[layer][0] for layer in layers}

    top: list[list[list[str]]] = [[] for _ in read_units]
    for layer in layers:
        # One transport per layer: the Jacobian moves to the device once.
        transported = lens.transport(activations.pop(layer).float(), layer)
        ranked = top_per_unit(transported, read_units, model.unembed, top_n)
        for u, ids in enumerate(ranked):
            top[u].append([decode(i) for i in ids])
    return LatentReadout(
        context,
        context_tokens,
        [unit[-1] for unit in read_units],
        layers,
        top,
        None if units is None else read_units,
    )


def read_span(
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    fmt: TranscriptFormat,
    kind: Kind,
    q: str,
    r: str | None = None,
    role: str | None = None,
    band: tuple[int, int] = BAND,
) -> LatentReadout:
    """Read out J's span for ``kind`` (DESIGN.md, Latent readout, Span), unit
    by unit, from Q on: Q and the generation prompt up to R, or up to D,
    with R's sentences as units if there is an R. ``role``: None for the
    advisor, or another role such as ``fmt.EDIT_ROLE``."""
    if kind == "reasoning":
        context = fmt.reasoning_prompt(q, role)
    else:
        context = fmt.decision_prompt(q, r, role=role)
    start = context.index(q)
    sentences = None
    if kind == "explanation" and r is not None:
        r_start = context.index(r, start + len(q))
        sentences = Span(r_start, r_start + len(r))
    span_units = units(model.tokenizer, context, start, sentences)
    readout = read_out(model, lens, context, band, units=span_units)
    readout.q_units = len(units(model.tokenizer, context, start, end=start + len(q)))
    return readout
