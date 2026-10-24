# CLAUDE.md

Library (`j_faithfulness`) plus marimo notebooks in `scripts/`: use the
Jacobian lens to test whether an LLM's rationales (reasoning traces and
explanations) are faithful to its latent reasoning.

- **[DESIGN.md](DESIGN.md):** method (v4, a discrimination task), status,
  next steps. Read first.
- **[README.md](README.md):** for users: the idea, install, notebooks, own
  contrasts.
- **[docs/archive/](docs/archive/):** earlier designs. The v2 pilot's code is
  at the git tag `v2-pilot`, the v3 probe's (IS/NOT, teacher-forced) at
  `v3-probe`.

## Layout

```
src/j_faithfulness/
  items.py      item format; DailyDilemmas adapter; dev and test samples
  formats.py    the Qwen3.5 transcript format; rebuild a transcript from a row
  contrasts.py  contrast roles as data; contrasts/*.yaml are the built-ins
  lens.py       load a model and its lens; read out a case; score A/B questions
  verbalize.py  readouts → J and the reversed J
  task.py       the discrimination task: questions, dilution, trials, the fit
scripts/
  demo.py         one item, one case, step by step (tutorial)
  experiments.py  one model, many items, one contrast: results JSON and charts
  generate.py     the transcript cache
```

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
- All work happens on `dev`; releases are merged into `main` and tagged.
- Model-free code is tested locally; GPU work (readouts, the questions,
  generating transcripts) runs on molab, which installs the library from
  GitHub, so push `dev` before a molab run.
- Results JSONs (`experiments_*.json`) are git-ignored.

## Code style

- Let names carry the meaning: a function or variable says what it is
  (`read_case`, `item_fits`, `log_odds_a`), so most lines need no
  comment.
- Comment the why, not the what: a design choice ("max, not mean: a peak at
  one word survives"), a pitfall, or a pointer to DESIGN.md. Never narrate
  the code.
- Small, pure functions: data in, data out, no hidden state, so the notebooks
  stay glue.
- Notebook cells stay flat: a dozen lines or so, nesting at most one level,
  no function definitions, no layout code. A cell over budget means a
  library function is missing.
- Charts: Altair only, off-the-shelf recipes.

## Prose style

- Plain words, short sentences, one idea per sentence. Each paragraph opens
  with its point.
- Say exactly what you mean, then stop: no hedging, intensifiers or filler.
  When something is uncertain, say once what and why.
- Define a term once, then use it consistently; don't vary words for style.
  A new word suggests a new thing.
