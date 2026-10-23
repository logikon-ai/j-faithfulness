# CLAUDE.md

Library (`j_faithfulness`) plus marimo notebooks in `scripts/`: use the
Jacobian lens to test whether an LLM's rationales (reasoning traces and
explanations) are faithful to its latent reasoning.

- **[DESIGN.md](DESIGN.md):** method (v4, a discrimination task), status,
  next steps. Read first.
- **[docs/archive/](docs/archive/):** earlier designs. The v2 pilot's code is
  at the git tag `v2-pilot`, the v3 probe's (IS/NOT, teacher-forced) at
  `v3-probe`.

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
- Model-free code is tested locally; GPU work (readout, the discrimination
  questions, generating the transcript cache) runs on molab.

## Code style

- Let names carry the meaning: a function or variable says what it is
  (`read_out`, `top_per_unit`, `log_odds_a`), so most lines need no
  comment.
- Comment the why, not the what: a design choice ("max, not mean: a peak at
  one word survives"), a pitfall, or a pointer to DESIGN.md. Never narrate
  the code.
- Small, pure functions: data in, data out, no hidden state, so the notebooks
  stay glue.

## Prose style

- Plain words, short sentences, one idea per sentence. Each paragraph opens
  with its point.
- Say exactly what you mean, then stop: no hedging, intensifiers or filler.
  When something is uncertain, say once what and why.
- Define a term once, then use it consistently; don't vary words for style.
  A new word suggests a new thing.
