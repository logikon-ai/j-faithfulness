# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "transformers>=5.5",
#     "jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e",
# ]
# ///
# torch is left out on purpose: molab preinstalls a CUDA build, and resolving
# torch from PyPI could replace it with one that doesn't support the GPU.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch

    return mo, torch


@app.cell
def _(mo):
    mo.md(r"""
    # M0 smoke test (molab, GPU)

    1. The GPU is visible and this torch build supports it (Blackwell is `sm_120`).
    2. Qwen3.5-4B and its pre-fitted Jacobian lens load.
    3. The jlens walkthrough readout reproduces.
    4. The chat template drops `<think>` from earlier assistant turns, so
       transcripts are raw strings.
    5. A first look at a readout at the injection point of a forced-choice
       question (protocol 1).
    """)
    return


@app.cell
def _(torch):
    assert torch.cuda.is_available(), "no CUDA GPU: attach one in the notebook specs"
    _cap = torch.cuda.get_device_capability()
    _arch = f"sm_{_cap[0]}{_cap[1]}"
    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": _arch,
        "arch_list": torch.cuda.get_arch_list(),
    }
    assert _arch in env["arch_list"], f"this torch build has no kernels for {_arch}"
    # A real kernel launch, not just the device query.
    (
        torch.ones(4, device="cuda", dtype=torch.bfloat16)
        @ torch.ones(4, device="cuda", dtype=torch.bfloat16)
    ).item()
    env
    return


@app.cell
def _(torch):
    import jlens
    import transformers

    MODEL_NAME = "Qwen/Qwen3.5-4B"
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)

    lens = jlens.JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens",
        filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
        revision="qwen-n1000",
    )
    {"n_layers": model.n_layers, "source_layers": lens.source_layers}
    return hf_model, lens, model, tokenizer


@app.cell
def _(mo):
    mo.md(r"""
    ## Walkthrough readout

    Two-hop question read out at the "boot" token. Expect the J-lens to
    surface Italy/euro-like tokens at middle layers where the logit lens is
    still noise.
    """)
    return


@app.cell
def _(lens, model, tokenizer):
    def top(logits, k=5):
        return [tokenizer.decode([t]) for t in logits.topk(k).indices]

    _prompt = "Fact: The currency used in the country shaped like a boot is"
    _layers = [
        model.n_layers // 4,
        model.n_layers // 2,
        model.n_layers // 4 * 3,
        model.n_layers - 2,
    ]
    _jl, _model_logits, _ = lens.apply(model, _prompt, layers=_layers, positions=[-2])
    _ll, _, _ = lens.apply(
        model, _prompt, layers=_layers, positions=[-2], use_jacobian=False
    )
    [
        {"layer": _l, "logit lens": top(_ll[_l][0]), "J-lens": top(_jl[_l][0])}
        for _l in _layers
    ] + [{"layer": "model", "logit lens": top(_model_logits[0]), "J-lens": ""}]
    return (top,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Chat template vs raw strings

    The template renders assistant turns before the last user turn without
    their `<think>` block. Expected: R is gone from the first rendering, and
    the raw string keeps it.
    """)
    return


@app.cell
def _(tokenizer):
    _msgs = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "<think>\nR\n</think>\n\nA"},
        {"role": "user", "content": "Explain."},
    ]
    templated = tokenizer.apply_chat_template(
        _msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    raw = (
        "<|im_start|>user\nQ?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nR\n</think>\n\nA<|im_end|>\n"
        "<|im_start|>user\nExplain.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    {"templated": templated, "raw": raw, "R kept by template": "R\n" in templated}
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Readout at the injection point (protocol 1)

    A forced-choice question, thinking off, read out at the last token of
    the generation prompt (where D is produced). Every second source layer,
    top 10 tokens. A first look for choosing the band in M1.
    """)
    return


@app.cell
def _(hf_model, lens, model, tokenizer, top, torch):
    _q = (
        "A town can fund either a new library or a new bus line, not both. "
        "Which should it fund? Answer with A (library) or B (bus line) only."
    )
    prompt_p1 = (
        f"<|im_start|>user\n{_q}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    _layers = lens.source_layers[::2]
    _jl, _model_logits, _ids = lens.apply(
        model, prompt_p1, layers=_layers, positions=[-1], max_seq_len=4096
    )
    _ab = tokenizer.convert_tokens_to_ids(["A", "B"])
    _p = torch.softmax(_model_logits[0].float(), -1)[_ab]
    with torch.no_grad():
        _gen = hf_model.generate(_ids, max_new_tokens=8, do_sample=False)
    readout_p1 = [
        {"layer": _l, "top 10": ", ".join(top(_jl[_l][0], 10))} for _l in _layers
    ]
    {
        "n_tokens": _ids.shape[-1],
        "P(A), P(B)": [round(x, 3) for x in _p.tolist()],
        "greedy reply": tokenizer.decode(_gen[0, _ids.shape[-1] :]),
        "readout": readout_p1,
    }
    return


if __name__ == "__main__":
    app.run()
