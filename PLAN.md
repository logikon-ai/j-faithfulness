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
- [ ] molab setup: find out how a notebook installs packages (probably inline
      script metadata); install `jlens` and this repo from git (needs the repo
      public, or auth); check that torch supports the Blackwell GPU.
      `notebooks/smoke.py` declares its dependencies as inline script metadata
      and leaves torch to molab's preinstalled CUDA build. Still to confirm on
      molab: that the metadata, including the git URL, is honoured. The repo
      goes public with the work on `dev`; molab installs this package from
      `git+https://github.com/logikon-ai/j-faithfulness.git@dev`, with a forced
      reinstall after each push.
- [ ] Smoke test on molab: load Qwen3.5-4B and its lens and reproduce the
      jlens walkthrough readout. `notebooks/smoke.py` is ready to run; it
      also checks the GPU arch and shows a first protocol-1 readout.
- [x] Check whether the Qwen3.5 chat template strips `<think>` content from
      earlier turns; work out the raw-string format. It does; the format is
      in DESIGN.md (Conversation protocols).

## M1. Pilot: case A in protocol 1
- [ ] `lens.py`: `read_at(model, lens, context, band, stride=2)` →
      `LatentReadout` (top tokens per selected layer at the injection point)
- [ ] `verbalize.py`: word-like mask; option to drop tokens already in Q;
      `verbalize(readout, k=10)` → J (one row per layer); `fence(J)` →
      `<latentthought>` block
- [ ] Inspect J on 3–5 example prompts; fix the band for Qwen3.5-4B
- [ ] `protocols.py`: raw-string transcript for protocol 1 (thinking off);
      forced-choice D (single token)
- [ ] `inject.py`: notice text; insertion at the end of the user turn
- [ ] `ablations.py`: own J, `drop_top(n)`, `drop_random(n)`,
      `replace_top(n, foils)`, `truncate(s)`
- [ ] `compare.py`: Brier score of D′ against D
- [ ] Notebook v1: Q → D → J → conditions (incl. no J, notice only) →
      contrast table and plot; small built-in item set (5–10 forced-choice
      questions)

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
