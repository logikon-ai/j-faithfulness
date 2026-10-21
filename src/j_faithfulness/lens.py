"""Load the model and its pre-fitted lens; read the lens out word by word."""

from __future__ import annotations

from dataclasses import dataclass

import jlens
import torch

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
LENS_REVISION = "qwen-n1000"

# lens.apply truncates from the right at max_seq_len, which would cut off the
# end of the context. Read-outs pass this and fail if the context reaches it.
MAX_SEQ_LEN = 32768


def load(model_name: str = MODEL_NAME):
    """Load Qwen3.5-4B (bf16, CUDA) and its lens.

    Returns ``(hf_model, model, lens)``: the HF model (for generation), its
    jlens wrapper, and the :class:`jlens.JacobianLens`.
    """
    import transformers

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    return hf_model, model, lens


@dataclass
class LatentReadout:
    """Lens read-out over a span of positions.

    ``top[i][j]`` are the decoded top tokens at ``positions[i]`` and
    ``layers[j]``, in rank order (index = rank), unfiltered;
    :func:`j_faithfulness.verbalize.verbalize` turns them into J.
    ``context_tokens`` is the whole context, decoded token by token.
    """

    context: str
    context_tokens: list[str]
    positions: list[int]
    layers: list[int]
    top: list[list[list[str]]]

    def in_band(self, lo: int, hi: int) -> LatentReadout:
        """The same read-out restricted to layers ``lo`` to ``hi`` inclusive."""
        keep = [j for j, layer in enumerate(self.layers) if lo <= layer <= hi]
        return LatentReadout(
            self.context,
            self.context_tokens,
            self.positions,
            [self.layers[j] for j in keep],
            [[ranked[j] for j in keep] for ranked in self.top],
        )


def readouts_to_json(
    readouts: dict[str, LatentReadout], top_n: int | None = None
) -> dict:
    """Read-outs as JSON-ready data, for tuning J offline. Tokens are stored
    once in a string table; ``top_n`` truncates each ranked list."""
    table: dict[str, int] = {}

    def index(s: str) -> int:
        return table.setdefault(s, len(table))

    out = {
        name: {
            "context": r.context,
            "context_tokens": r.context_tokens,
            "positions": r.positions,
            "layers": r.layers,
            "top": [
                [[index(t) for t in ranked[:top_n]] for ranked in layers]
                for layers in r.top
            ],
        }
        for name, r in readouts.items()
    }
    return {"strings": list(table), "readouts": out}


def readouts_from_json(data: dict) -> dict[str, LatentReadout]:
    """Inverse of :func:`readouts_to_json`."""
    strings = data["strings"]
    return {
        name: LatentReadout(
            r["context"],
            r["context_tokens"],
            r["positions"],
            r["layers"],
            [
                [[strings[i] for i in ranked] for ranked in layers]
                for layers in r["top"]
            ],
        )
        for name, r in data["readouts"].items()
    }


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


@torch.no_grad()
def read_out(
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    context: str,
    band: tuple[int, int],
    positions: list[int] | None = None,
    skip: str = "",
    top_n: int = 200,
) -> LatentReadout:
    """Read out the lens at every layer of ``band`` (inclusive) and at
    ``positions`` (default: every word end of ``context`` after the leading
    text ``skip``, e.g. the opening template of the user turn).

    ``top_n`` raw tokens are kept per position and layer: enough for word-like
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
    if positions is None:
        start = skipped_tokens(context_tokens, skip)
        positions = [p for p in word_ends(context_tokens) if p >= start]
    positions = [p % n for p in positions]

    lens_logits, _, _ = lens.apply(
        model, context, layers=layers, positions=positions, max_seq_len=MAX_SEQ_LEN
    )
    ids = {layer: lens_logits[layer].topk(top_n).indices.tolist() for layer in layers}
    top = [
        [[decode(i) for i in ids[layer][p]] for layer in layers]
        for p in range(len(positions))
    ]
    return LatentReadout(context, context_tokens, positions, layers, top)
