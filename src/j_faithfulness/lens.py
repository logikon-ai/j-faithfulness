"""Load the model and its pre-fitted lens; read the lens out at the injection
point."""

from __future__ import annotations

from dataclasses import dataclass

import jlens
import torch

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
LENS_REVISION = "qwen-n1000"

# lens.apply truncates from the right at max_seq_len, which would cut off the
# injection point. Read-outs pass this and fail if the context reaches it.
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
    """Lens read-out at one position: the top tokens per layer, in rank order.

    ``tokens[i]`` are the decoded top ``len(tokens[i])`` tokens of
    ``layers[i]``, unfiltered; :func:`j_faithfulness.verbalize.verbalize`
    selects the word-like ones.
    """

    layers: list[int]
    tokens: list[list[str]]
    token_ids: list[list[int]]
    n_context_tokens: int


def band_layers(band: tuple[int, int], stride: int = 2) -> list[int]:
    """Layers ``band[0], band[0] + stride, ...`` up to ``band[1]`` inclusive."""
    lo, hi = band
    return list(range(lo, hi + 1, stride))


@torch.no_grad()
def read_at(
    model: jlens.LensModel,
    lens: jlens.JacobianLens,
    context: str,
    band: tuple[int, int],
    stride: int = 2,
    top_n: int = 200,
) -> LatentReadout:
    """Read out the lens at the last token of ``context`` (the injection point)
    for every ``stride``-th layer in ``band``.

    ``top_n`` raw tokens are kept per layer: enough to leave ten or so once
    non-word tokens are filtered out.
    """
    layers = band_layers(band, stride)
    lens_logits, _, input_ids = lens.apply(
        model, context, layers=layers, positions=[-1], max_seq_len=MAX_SEQ_LEN
    )
    n = input_ids.shape[-1]
    if n >= MAX_SEQ_LEN:
        raise ValueError(f"context reaches MAX_SEQ_LEN={MAX_SEQ_LEN} tokens")
    tokens, token_ids = [], []
    for layer in layers:
        ids = lens_logits[layer][0].topk(top_n).indices.tolist()
        token_ids.append(ids)
        tokens.append(
            [
                model.tokenizer.decode([i], clean_up_tokenization_spaces=False)
                for i in ids
            ]
        )
    return LatentReadout(layers, tokens, token_ids, n)
