# Plan: library + demo notebook

Goal: a marimo notebook that imports `j_faithfulness` and runs the stability
and uptake tests for cases A, B and C on a small set of forced-choice
questions (everyday dilemmas from DailyDilemmas), as scoped in
[DESIGN.md](DESIGN.md) (Test logic → Scope). M1 is the pilot.

## Where things run
- **Locally (no GPU):** everything that doesn't need the model: `verbalize`,
  `inject`, `protocols` (string building), `compare`. Tested with mock
  `LatentReadout`s or the tiny model in `tests/tiny.py` (vendored from
  jlens, which doesn't ship its tests).
- **[molab](https://marimo.io/blog/reintroducing-molab) (GPU):** running the
  notebook. GPU-dependent code is prototyped in notebook cells, then moved
  into the package once it settles.
- **molab setup:** a notebook declares its dependencies as inline script
  metadata, and molab installs them itself, including `jlens` from its git
  URL and this package from
  `git+https://github.com/logikon-ai/j-faithfulness.git@dev` (the repo is
  public; work stays on `dev`). torch is left out: molab's preinstalled CUDA
  build (torch 2.11, CUDA 13.0, `sm_120` kernels) supports the Blackwell GPU
  and survives the install. A running session needs a forced reinstall after
  each push.
- **molab constraints:** sessions last at most 12 hours and shut down after 90
  minutes idle. Storage is temporary, so the model and lens are downloaded
  again each session. Notebooks are public (but not discoverable), so no
  tokens or secrets go in them.

## Done
- Scaffolding: src layout, uv, ruff, pytest; molab setup and smoke test (the
  jlens walkthrough read-out reproduced on Qwen3.5-4B).
- Raw-string transcripts (the chat template strips `<think>` from earlier
  turns; format in DESIGN.md, Conversation protocols).
- `lens.py`: `load()`, `read_out()` over a span, word by word.
- `verbalize.py`: J by TF × IDF, tuned on eight one-hop items (band 8–21,
  `min_tf` 1.0, k 5, English words only): J surfaces 7 of 8 intermediates.
- `protocols.py`: protocol 1 up to D; `option_probs`, `decision`.

## M1. Pilot: case A in protocol 1
- [x] **Notebook skeleton** (`notebooks/demo.py`): every section of
      DESIGN.md (Scope) in place — selection with a run button and verdict
      thresholds, walkthrough on one item, A1, A2, B, C3, C4, summary
      heatmap. The walkthrough is plain code, one cell per step (item →
      prompt → D → read-out and J → reference concepts in J → injected
      prompt → D′), each showing its own output. The case sections share
      helpers in one cell (`section`, `scatter`, `answer_chart`), so each case
      cell is a single call. `cases.py` holds what the sections call: `CASES`
      (title, transcript sketch, milestone), `run(lm, item, case)` → outcome
      per condition, `stability`, `uptake`, `verdict`. `run` raises
      `NotImplementedError` for unbuilt cases, shown as a "not yet
      implemented (Mn)" callout. A1 runs; dry-run on CPU with the tiny
      model. M2 fills C3; M3 fills A2, B and C4; M4 adds the domain view.
- [x] Smoke test of the skeleton on molab: A1 on the one-hop items with
      Qwen3.5-4B. The notebook runs through and plots.
- [x] `inject.py`: notice, empty block, follow and oppose instructions
- [x] `compare.py`: JSD (stability and uptake)
- [x] `items.py`: the shared item format (`Item`: id, Q, options, actions,
      reference concepts, domain; DESIGN.md, Items and adapters) and the
      DailyDilemmas adapter. It reads the dataset's CSV at a pinned revision
      and turns each dilemma (one row per option) into a binary item:
      Q = the situation, A = to do, B = not to do, the values as reference
      concepts, `topic_group` as the domain; the consequences stay out of Q.
      `split` gives fixed, disjoint dev and test halves. The notebook selects
      by split, number of items and domains; the walkthrough shows which of
      each option's values J touches. The one-hop items are gone.
- [ ] Walkthrough section: read-out check on the dev items with thinking
      off. Does J surface the annotated values (overlap with each option's
      values, and with D's)? Revisit band and thresholds if needed.
- [ ] **Checkpoint:** if J carries no value content with thinking off, the
      pilot moves to thinking on (protocol 2, J read out before R) and M3's
      protocol work comes first.
- [ ] Run stability and uptake on molab; read stability against uptake
      (DESIGN.md, the reading table). Then confirm on the test items.

## M2. Case C (protocol 3)
- [ ] `protocols.py`: explanation turn (thinking off); force the original D
- [ ] `compare.py`: lexical distance for E vs E′ (TF-IDF cosine to start);
      teacher-forced log P(E | condition); noise floor from resampling the
      original
- [ ] Notebook: case C section, stability and uptake on E
- [ ] **Checkpoint:** if lexical distance and teacher-forced log P don't
      separate the conditions on E, reconsider the fallback measures
      (embedding similarity, NLI, LLM judge)

## M3. Case B (protocols 2 and 4)
- [ ] `protocols.py`: thinking on; locate R and D; teacher-force R
- [ ] `inject.py`: injection right before R; splitting an assistant turn with a
      short user turn (also needed for case C in protocol 4)
- [ ] `compare.py`: R vs R′ (lexical; per-token teacher-forced log P); total
      vs direct effect on D
- [ ] Notebook: case B section

## M4. Domains
- [ ] Notebook summary: domains × subcases (share faithful, unfaithful,
      uninformative), first over DailyDilemmas' topic groups
- [ ] A second adapter with items from another domain

## M5. Docs
- [ ] README: what the method is, install, import from another repo, link to
      the notebook
