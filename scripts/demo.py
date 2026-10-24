# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
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
    import datasets
    import marimo as mo

    from j_faithfulness.contrasts import load_contrast
    from j_faithfulness.formats import QWEN35
    from j_faithfulness.lens import LENSES, load, log_odds_a, read_case
    from j_faithfulness.task import question, spread_rows
    from j_faithfulness.verbalize import j_and_reversed, rows_text

    return (
        LENSES,
        QWEN35,
        datasets,
        j_and_reversed,
        load,
        load_contrast,
        log_odds_a,
        mo,
        question,
        read_case,
        rows_text,
        spread_rows,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # j-faithfulness, step by step

    Does a model's explanation respond to the deliberation in its latent
    state? We read that deliberation out with the Jacobian lens, as rows of
    key words, and ask the model which of two readouts belongs to its
    explanation.
    """)
    return


@app.cell
def _(datasets):
    MODEL_ID = "Qwen/Qwen3.5-4B"
    DATASET = "ggbetz/cached-reasoning-records"
    SHA = "2d98d41c2e0395eeb871cc3782de24975c54f5d0"
    ITEM_ID = "dd-40145"

    records = datasets.load_dataset(DATASET, revision=SHA, split="train")
    # Each item has two transcripts; take the one with thinking on.
    transcript = next(
        row
        for row in records
        if row["metadata"]["model_id"] == MODEL_ID
        and row["id"] == ITEM_ID
        and row["r"] is not None
    )

    print("Q:", transcript["q"], sep="\n")
    print("\nR:", transcript["r"], sep="\n")
    print("\nD:", transcript["d"])
    print("\nE:", transcript["e"], sep="\n")
    return MODEL_ID, records, transcript


@app.cell
def _(LENSES, MODEL_ID, load, load_contrast):
    hf_model, model, lens = load(MODEL_ID)
    band = LENSES[MODEL_ID].band
    contrast_roles = load_contrast("editor")

    print(f"{MODEL_ID}: {model.n_layers} layers, lens band {band}")
    for role in contrast_roles:
        print(f"\n{role.name}:\n{role.template}")
    return band, contrast_roles, hf_model, lens, model


@app.cell
def _(band, contrast_roles, lens, model, read_case, transcript):
    # The explanation case: read Q and R up to D, under the advisor and
    # under each contrast role.
    advisor, contrast = read_case(
        model,
        lens,
        band,
        contrast_roles,
        transcript["q"],
        "explanation",
        transcript["r"],
    )

    # Top words at the last unit, right before D, at the band's middle layer.
    middle = len(advisor.top[-1]) // 2
    print("advisor:", advisor.top[-1][middle][:15])
    print("editor: ", contrast[0].top[-1][middle][:15])
    return advisor, contrast


@app.cell
def _(advisor, contrast, j_and_reversed, rows_text):
    j, reversed_j = j_and_reversed(advisor, contrast)

    print("J, what deciding adds over the editor:", rows_text(j), sep="\n")
    print("\nReversed J, what the editor adds:", rows_text(reversed_j), sep="\n")
    return j, reversed_j


@app.cell
def _(mo):
    mo.md(r"""
    ## Which readout belongs to the explanation?

    The model sees its explanation E and two readouts: J and the reversed J.
    Both were read over the same tokens, so topic can't tell them apart.
    """)
    return


@app.cell
def _(j, reversed_j, spread_rows):
    # Cut both candidates to one length, keeping evenly spaced rows: the
    # rows read right before D sit at the end.
    n_rows = min(len(j), len(reversed_j))
    right = spread_rows(j, n_rows)
    wrong = spread_rows(reversed_j, n_rows)

    print(f"{n_rows} rows per candidate")
    return right, wrong


@app.cell
def _(QWEN35, question, right, transcript, wrong):
    # Each question in both orders: averaging cancels a bias towards A or B.
    right_first = QWEN35.answer_prompt(
        question("explanation", transcript["e"], right, wrong)
    )
    right_second = QWEN35.answer_prompt(
        question("explanation", transcript["e"], wrong, right)
    )

    print(right_first)
    return right_first, right_second


@app.cell
def _(hf_model, log_odds_a, model, right_first, right_second):
    first, second = log_odds_a(hf_model, model.tokenizer, [right_first, right_second])
    # log-odds of the right answer: A when it comes first, B when second.
    own_x = (first - second) / 2

    print(f"log-odds of the right readout, with its own E: {own_x:+.2f}")
    return (own_x,)


@app.cell
def _(mo):
    mo.md(r"""
    ## The control: another item's explanation

    J may simply sound more like deciding than the reversed J does, whatever
    E says. The same question with another item's E measures that
    preference. The fit is what E's own text adds over it.
    """)
    return


@app.cell
def _(MODEL_ID, records):
    FOREIGN_ID = "dd-28728"

    foreign = next(
        row
        for row in records
        if row["metadata"]["model_id"] == MODEL_ID
        and row["id"] == FOREIGN_ID
        and row["r"] is not None
    )

    print("Foreign E:", foreign["e"], sep="\n")
    return (foreign,)


@app.cell
def _(QWEN35, foreign, hf_model, log_odds_a, model, question, right, wrong):
    foreign_prompts = [
        QWEN35.answer_prompt(question("explanation", foreign["e"], right, wrong)),
        QWEN35.answer_prompt(question("explanation", foreign["e"], wrong, right)),
    ]
    first_f, second_f = log_odds_a(hf_model, model.tokenizer, foreign_prompts)
    foreign_x = (first_f - second_f) / 2

    print(f"log-odds of the right readout, with a foreign E: {foreign_x:+.2f}")
    return (foreign_x,)


@app.cell
def _(foreign_x, own_x):
    fit = own_x - foreign_x

    print(f"fit on this item: {own_x:+.2f} − ({foreign_x:+.2f}) = {fit:+.2f}")
    return


if __name__ == "__main__":
    app.run()
