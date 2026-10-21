# Plan: library + demo notebook

Goal: a marimo notebook that imports `j_faithfulness` and demonstrates cases A,
B and C on a small set of forced-choice questions, as scoped in
[DESIGN.md](DESIGN.md) (Test logic → Scope). We build it in milestones; M1 is
the pilot.

## Where things run
- **Locally (no GPU):** everything that doesn't need the model: `verbalize`,
  `ablations`, `inject`, `protocols` (string building), `compare`. Tested
  with mock `LatentReadout`s or a tiny model (jlens ships `tests/tiny.py`).
- **[molab](https://marimo.io/blog/reintroducing-molab) (GPU):** smoke test,
  inspecting J and choosing the band, running the notebook. While exploring,
  GPU-dependent code is prototyped in notebook cells, then moved into the
  package once it settles.
- **molab constraints:** sessions last at most 12 hours and shut down after 90
  minutes idle. Storage is temporary, so the model and lens are downloaded
  again each session. Notebooks are public (but not discoverable), so no
  tokens or secrets go in them. Code moves from local to molab via push to
  GitHub, then reinstall.

## M0. Scaffolding
- [x] `pyproject.toml` (src layout, uv). Dependencies: `jlens` from git,
      `torch`, `transformers>=5.5`, `marimo`. Dev dependencies: `pytest`, `ruff`.
- [x] `src/j_faithfulness/__init__.py`, `tests/`, `notebooks/`
      (`tests/tiny.py` is vendored from jlens, which doesn't ship its tests)
- [x] molab setup: find out how a notebook installs packages (probably inline
      script metadata); install `jlens` and this repo from git (needs the repo
      public, or auth); check that torch supports the Blackwell GPU.
      `notebooks/smoke.py` declares its dependencies as inline script metadata
      and leaves torch to molab's preinstalled CUDA build (torch 2.11, CUDA
      13.0, with `sm_120` kernels: the Blackwell GPU works). molab installs
      the metadata's dependencies itself, including `jlens` from its git URL,
      and this package from
      `git+https://github.com/logikon-ai/j-faithfulness.git@dev` (the repo is
      public; work stays on `dev`). The preinstalled torch survives the
      install. A session that's still running needs a forced reinstall after
      each push.
- [x] Smoke test on molab: load Qwen3.5-4B and its lens and reproduce the
      jlens walkthrough readout. `notebooks/smoke.py`: at the "boot" token the
      J-lens shows boots/heel at layer 8 and Italy at layer 16, where the
      logit lens is still noise. First protocol-1 readout is under Band in
      DESIGN.md.
- [x] Check whether the Qwen3.5 chat template strips `<think>` content from
      earlier turns; work out the raw-string format. It does; the format is
      in DESIGN.md (Conversation protocols).

## M1. Pilot: case A in protocol 1
- [x] `lens.py`: `read_out(model, lens, context, band, positions=None)` →
      `LatentReadout` (top tokens at every word of the span and every layer
      of the band); `load()` for Qwen3.5-4B and its lens
- [x] `verbalize.py`: TF × IDF over the read-out's ranked lists (IDF from
      the other positions); word-like, Latin script, TF minimum, context echo
      incl. prefix variants; `verbalize(readout, k=10)` → J (one row per
      position); `fence(J)` → `<latentthought>` block
- [x] First read-outs at the single token before D on five one-shot items
      (`notebooks/band.py`, since replaced): J held only response format and
      the answer. This led to the span read-out and to decision prompts with
      a hidden step (DESIGN.md, Construct).
- [x] Validate the read-out and fix the band and thresholds on eight one-hop
      decision items with known intermediates plus one one-shot control
      (`notebooks/readout.py`). J over the span surfaces 7 of 8
      intermediates, J at the last token none. Tuned offline on the
      downloaded read-outs (`notebooks/tune.py`): band 8–21, `min_tf` 1.0,
      k 5, English words only; J has 12–54 words. IDF over a batch of
      prompts didn't help.
- [x] `protocols.py`: raw-string transcript for protocol 1 (thinking off);
      forced-choice D (single token): `option_probs` (not renormalized, so
      mass leaving the options shows), `decision`
- [x] `inject.py`: notice text; insertion at the end of the user turn
- [x] `ablations.py`: `swap(mapping)`, `swap_random` (the same foils in
      place of random words), `drop_top(n)`, `drop_random(n)`,
      `replace_top(n, foils)`, `truncate(s)`; salience is TF × IDF
- [x] `compare.py`: Brier score of D′ against D; log P(D)
- [x] `items.py`: the one-hop items, with foils that imply the other answer
- [ ] Notebook v1 (`notebooks/demo.py`): Q → D → J → conditions (no J,
      notice only, own J, intermediate → foil vs random word → foil, drop
      top 3 vs drop random 3) → contrast table and plot. Written and
      dry-run on CPU; not yet run on molab.

## M2. Case C (protocol 3), with agreement with A
- [ ] `protocols.py`: explanation turn (thinking off); force the original D
- [ ] `compare.py`: text distances for E vs E′ (start with one or two of the
      options); noise floor from resampling under own J
- [ ] Single-item edits; rank agreement between the D shift and the E distance
- [ ] Notebook: case C section

## M3. Case B (protocols 2 and 4)
- [ ] `protocols.py`: thinking on; locate R and D; teacher-force R
- [ ] `inject.py`: injection right before R; splitting an assistant turn with a
      short user turn (also needed for case C in protocol 4)
- [ ] `compare.py`: R vs R′ (text distances; per-token teacher-forced log P);
      total vs direct effect on D
- [ ] Notebook: case B section

## M4. Full notebook
- [ ] Opposite-answer J as the positive control in every case
- [ ] `ablations.py`: foreign J, same-answer/different-reasons J, mix
      (dose-response), shuffles (within and across rows), framing variants
- [ ] Notebook: one section showing the rest of the ladder once
- [ ] Polish: narrative markdown cells, consistent plots

## M5. Docs
- [ ] README: what the method is, install, import from another repo, link to
      the notebook
- [ ] CLAUDE.md: points to DESIGN.md and PLAN.md; conventions
