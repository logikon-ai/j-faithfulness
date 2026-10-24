# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "altair",
#     "datasets",
#     "transformers>=5.5",
#     "flash-linear-attention",
#     "jlens @ git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e",
#     "j-faithfulness @ git+https://github.com/logikon-ai/j-faithfulness.git@dev",
# ]
# ///
# torch is left out on purpose: molab preinstalls a CUDA build.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from dataclasses import asdict

    import altair as alt
    import datasets
    import marimo as mo

    from j_faithfulness import task
    from j_faithfulness.contrasts import contrast_name, load_contrast
    from j_faithfulness.formats import QWEN35
    from j_faithfulness.lens import (
        LENS_REPO,
        LENS_REVISION,
        LENSES,
        load,
        log_odds_a,
        read_case,
    )
    from j_faithfulness.verbalize import j_and_reversed, rows_text

    return (
        LENSES,
        LENS_REPO,
        LENS_REVISION,
        QWEN35,
        alt,
        asdict,
        contrast_name,
        datasets,
        j_and_reversed,
        json,
        load,
        load_contrast,
        log_odds_a,
        mo,
        read_case,
        rows_text,
        task,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # j-faithfulness of one model, at one contrast

    Runs the discrimination task for one model over many items, at one
    contrast, and reports the fit per case. For a walkthrough of a single
    item, see `demo.py`; for the method, DESIGN.md.

    Set `MODEL_ID` and `CONTRAST` below (a built-in name or the path of your
    own YAML). The run writes `experiments_{model}_{contrast}.json` with the
    trials and a record of the run. It needs a GPU.

    The model sees a rationale X and two readouts and answers A or B. The
    right candidate is always X's own J. The conditions:

    | condition | wrong candidate | X shown | measures |
    |---|---|---|---|
    | foreign | another item's J | own | topic and more: the calibration |
    | reversed | the reversed J | own | fit beyond topic |
    | reversed, foreign X | the reversed J | another item's | register: the control |

    The fit is reversed minus reversed, foreign X, per item, at full signal.
    """)
    return


@app.cell
def _(contrast_name):
    MODEL_ID = "Qwen/Qwen3.5-4B"
    DATASET = "ggbetz/cached-reasoning-records"
    SHA = "2d98d41c2e0395eeb871cc3782de24975c54f5d0"
    # A built-in contrast's name, or the path of one's own YAML file.
    CONTRAST = "editor"
    N_ITEMS = 50
    DOSES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    # Distractor and filler pairs per case: different pairs average out how
    # unlike two particular items are.
    DRAWS = 4
    # Cases whose J or reversed J has fewer rows are left out: the
    # candidates are cut to the shorter one, and a few rows say little.
    MIN_ROWS = 5
    # One file per model and contrast, so a run doesn't overwrite another's.
    OUT = f"experiments_{MODEL_ID.split('/')[-1]}_{contrast_name(CONTRAST)}.json"
    return CONTRAST, DATASET, DOSES, DRAWS, MIN_ROWS, MODEL_ID, N_ITEMS, OUT, SHA


@app.cell
def _(DATASET, MODEL_ID, N_ITEMS, SHA, datasets, task):
    records = datasets.load_dataset(DATASET, revision=SHA, split="train")
    rows = [row for row in records if row["metadata"]["model_id"] == MODEL_ID]
    cases = task.cases(rows, MODEL_ID, N_ITEMS)
    # Read with the weights that wrote the transcripts.
    revisions = {row["metadata"]["model_revision"] for row in rows}
    assert len(revisions) == 1, f"transcripts from several revisions: {revisions}"
    (model_revision,) = revisions

    print(f"{len({c['id'] for c in cases})} items, {len(cases)} cases")
    print(f"{MODEL_ID} at {model_revision}")
    return cases, model_revision


@app.cell
def _(CONTRAST, LENSES, MODEL_ID, load, load_contrast, model_revision):
    band = LENSES[MODEL_ID].band
    hf_model, model, lens = load(MODEL_ID, model_revision)
    roles = load_contrast(CONTRAST)

    print(f"Load {MODEL_ID} ({model.n_layers} layers) with its Jacobian lens.")
    print(f"Contrast {CONTRAST!r}: {', '.join(role.name for role in roles)}.")
    return band, hf_model, lens, model, roles


@app.cell
def _(band, cases, j_and_reversed, lens, mo, model, read_case, roles):
    read = []
    for _c in mo.status.progress_bar(cases, title="Read out"):
        _advisor, _contrast = read_case(
            model, lens, band, roles, _c["q"], _c["kind"], _c["r"], _c["options"]
        )
        _j, _reversed_j = j_and_reversed(_advisor, _contrast)
        read.append({**_c, "j": _j, "reversed": _reversed_j})

    print(f"Read out {len(read)} cases as advisor and under the contrast.")
    return (read,)


@app.cell
def _(read, rows_text):
    _case = next(c for c in read if c["kind"] == "explanation" and c["r"])

    print(f"{_case['id']}, explanation, thinking on. J, what deciding adds:\n")
    print(rows_text(_case["j"]))
    print("\nReversed J, what the contrast adds:\n")
    print(rows_text(_case["reversed"]))
    return


@app.cell
def _(MIN_ROWS, read):
    kept = [c for c in read if min(len(c["j"]), len(c["reversed"])) >= MIN_ROWS]

    print(f"Rows per J; {len(kept)} of {len(read)} cases have ≥ {MIN_ROWS} in both:")
    [
        {
            "id": c["id"],
            "case": c["case"],
            "J": len(c["j"]),
            "reversed": len(c["reversed"]),
        }
        for c in read
    ]
    return (kept,)


@app.cell
def _(DOSES, DRAWS, kept, task):
    trials = task.trials(kept, DOSES, DRAWS)

    print(f"{len(trials)} trials, two questions each.")
    return (trials,)


@app.cell
def _(QWEN35, hf_model, log_odds_a, mo, model, task, trials):
    # All questions in one batched pass, right candidate first then second.
    prompts = [QWEN35.answer_prompt(t[o]) for t in trials for o in ("first", "second")]
    with mo.status.progress_bar(total=len(set(prompts)), title="Ask") as _bar:
        scores = log_odds_a(
            hf_model, model.tokenizer, prompts, lambda n: _bar.update(increment=n)
        )
    results = task.scored(trials, scores)

    print(f"Asked {len(set(prompts))} distinct questions for {len(trials)} trials.")
    return (results,)


@app.cell
def _(
    CONTRAST,
    DATASET,
    DOSES,
    DRAWS,
    LENSES,
    LENS_REPO,
    LENS_REVISION,
    MIN_ROWS,
    MODEL_ID,
    N_ITEMS,
    QWEN35,
    SHA,
    asdict,
    band,
    contrast_name,
    model_revision,
    roles,
    task,
):
    # What produced the trials (DESIGN.md, The record).
    run = {
        "model_id": MODEL_ID,
        "model_revision": model_revision,
        "contrast": {
            "name": contrast_name(CONTRAST),
            "roles": [asdict(r) for r in roles],
        },
        "format": QWEN35.name,
        "lens": {
            "repo": LENS_REPO,
            "revision": LENS_REVISION,
            "file": LENSES[MODEL_ID].file,
            "band": band,
        },
        "dataset": {"repo": DATASET, "sha": SHA, "n_items": N_ITEMS},
        "doses": DOSES,
        "draws": DRAWS,
        "min_rows": MIN_ROWS,
        "seed_rule": task.SEED_RULE,
    }
    run
    return (run,)


@app.cell
def _(OUT, json, kept, results, run):
    # The cases hold the J's, so the trials can be rebuilt without a GPU.
    with open(OUT, "w") as _f:
        json.dump({"run": run, "cases": kept, "trials": results}, _f)

    print(f"Wrote the run, {len(kept)} cases and {len(results)} trials to {OUT}.")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Results

    The charts show means over items with ± 1 standard error. Each question
    is asked in both orders and the log-odds of the right answer are
    averaged, which cancels a bias towards A or B. Both candidates are
    diluted the same way: a fraction p of their rows is replaced by a third
    item's J. The signal fraction 1 − p sets the difficulty; at 0 both
    candidates are the same and the answer is at chance.
    """)
    return


@app.cell
def _(results, task):
    # Per-item means over draws: the charts take standard errors over items,
    # and a chart of every trial outgrows marimo's output limit.
    means = task.item_means(results)
    fits = task.item_fits(means)
    return fits, means


@app.cell
def _(alt, fits):
    _full = alt.Chart(alt.Data(values=fits)).transform_filter("datum.signal == 1")
    _zero = _full.mark_rule(strokeDash=[4, 4]).encode(x=alt.datum(0))
    _bars = _full.mark_errorbar(extent="stderr").encode(
        x=alt.X("fit:Q", title="fit (log-odds)"), y=alt.Y("case:N", title=None)
    )
    _points = _full.mark_point(filled=True).encode(x="mean(fit):Q", y="case:N")

    (_zero + _bars + _points).properties(
        width=360, title="The fit at full signal, ± 1 SE over items:"
    )
    return


@app.cell
def _(results, task):
    # The fit at full signal: reversed minus reversed with a foreign X,
    # paired per item; mean and standard error over items.
    task.fit(results)
    return


@app.cell
def _(alt, fits):
    _base = alt.Chart(alt.Data(values=fits)).encode(
        x=alt.X("signal:Q", title="signal fraction (1 − p)"),
        color=alt.Color("case:N", title=None),
    )
    _band = _base.mark_errorband(extent="stderr").encode(
        y=alt.Y("fit:Q", title="fit (log-odds)")
    )
    _line = _base.mark_line(point=True).encode(y="mean(fit):Q")

    (_band + _line).properties(
        width=360, height=220, title="The fit as the candidates are diluted, ± 1 SE:"
    )
    return


@app.cell
def _(alt, means, task):
    # Foreign in grey: it is the calibration, not the measurement.
    condition_colors = alt.Scale(
        domain=list(task.CONDITIONS), range=["#bab0ac", "#4c78a8", "#f58518"]
    )
    _base = alt.Chart(alt.Data(values=means)).encode(
        x=alt.X("signal:Q", title="signal fraction (1 − p)"),
        color=alt.Color("condition:N", scale=condition_colors, title=None),
    )
    _band = _base.mark_errorband(extent="stderr").encode(
        y=alt.Y("log_odds:Q", title="log-odds of the right answer")
    )
    _line = _base.mark_line(point=True).encode(y="mean(log_odds):Q")

    (_band + _line).properties(width=220, height=180).facet(
        column=alt.Column("case:N", title=None)
    ).properties(title="Log-odds per condition, ± 1 SE over items:")
    return (condition_colors,)


@app.cell
def _(alt, condition_colors, means):
    (
        alt.Chart(alt.Data(values=means))
        .mark_line(point=True)
        .encode(
            x=alt.X("signal:Q", title="signal fraction (1 − p)"),
            y=alt.Y(
                "mean(correct):Q", title="accuracy", scale=alt.Scale(domain=[0, 1])
            ),
            color=alt.Color("condition:N", scale=condition_colors, title=None),
        )
        .properties(width=220, height=180)
        .facet(column=alt.Column("case:N", title=None))
        .properties(title="Accuracy per condition:")
    )
    return


if __name__ == "__main__":
    app.run()
