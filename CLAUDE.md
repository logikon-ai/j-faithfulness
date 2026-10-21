# CLAUDE.md

Library (`j_faithfulness`) plus one marimo demo notebook: use the Jacobian lens
to test the faithfulness (reason-responsiveness) of an LLM's decisions and
rationales.

- **[DESIGN.md](DESIGN.md):** method, test logic, open questions. Read first.
- **[PLAN.md](PLAN.md):** milestones and where things run (local vs molab GPU).
  Tick items off as they're done.

## jlens reference

[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)
(package `jlens`, unmaintained reference implementation; Python ≥3.10,
`transformers>=5.5`). Install from git; it's not on PyPI as far as we know.

```python
import jlens, torch, transformers

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B", dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = jlens.from_hf(hf, tok)          # LensModel; model.n_layers

lens = jlens.JacobianLens.from_pretrained(
    "neuronpedia/jacobian-lens",
    filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    revision="qwen-n1000",
)                                        # lens.source_layers

lens_logits, model_logits, input_ids = lens.apply(
    model, prompt, layers=[...], positions=[-1])
# lens_logits: {layer: [n_positions, vocab]}; model_logits: [n_positions, vocab]
# use_jacobian=False gives the plain logit lens; max_seq_len defaults to 512
```

- `lens.apply` takes a plain string, which suits our raw-string transcripts.
- `jlens.vis.compute_slice(model, lens, prompt, layer_stride=2,
  mask_display=True)` builds the layer × position view; `mask_display`
  restricts the display to word-like tokens.
- The jlens repo has `assets/qwen_gloss.json.gz` (English glosses for Qwen's
  CJK tokens) and `tests/tiny.py` (a tiny model for tests without a GPU).

## Conventions

- src layout, uv, ruff, pytest.
- All work happens on `dev`. `main` holds only a work-in-progress README until
  the first release, when `dev` is merged into it.
- Model-free code is tested locally; GPU work runs on molab (see PLAN.md).
