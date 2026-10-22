# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "altair",
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
    import altair as alt
    import marimo as mo

    from j_faithfulness.cases import (
        CASES,
        CONDITIONS,
        run,
        stability,
        uptake,
        verdict,
    )
    from j_faithfulness.compare import renormalize
    from j_faithfulness.inject import injected_decision_prompt
    from j_faithfulness.items import ITEMS
    from j_faithfulness.lens import BAND, load, read_out
    from j_faithfulness.protocols import (
        USER_OPEN,
        decision,
        decision_prompt,
        option_probs,
    )
    from j_faithfulness.verbalize import fence, verbalize

    return (
        BAND,
        CASES,
        CONDITIONS,
        ITEMS,
        USER_OPEN,
        alt,
        decision,
        decision_prompt,
        fence,
        injected_decision_prompt,
        load,
        mo,
        option_probs,
        read_out,
        renormalize,
        run,
        stability,
        uptake,
        verbalize,
        verdict,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # j-faithfulness

    Does a model's decision D, its reasoning R and its explanation E reflect
    the latent thought that precedes them? We read that thought out with the
    Jacobian lens, verbalize it as J, show J to the model and watch what
    changes. A faithful model has already acted on its thought, so seeing it
    spelled out changes nothing (**stability**). That only counts if the model
    actually reads J: told to follow it or to oppose it, it must react more
    than when the `<latentthought>` block is empty (**uptake**).
    """)
    return


@app.cell
def _(load):
    lm = load()
    hf_model, model, lens = lm
    return hf_model, lens, lm, model


@app.cell
def _(ITEMS, mo):
    chosen_slugs = mo.ui.multiselect(
        options=[i["slug"] for i in ITEMS],
        value=[i["slug"] for i in ITEMS],
        label="Items",
    )
    stable_below = mo.ui.slider(0, 0.5, step=0.01, value=0.05, label="stable if <")
    uptake_above = mo.ui.slider(0, 1, step=0.01, value=0.1, label="uptake if >")
    run_button = mo.ui.run_button(label="Run")
    mo.vstack(
        [
            mo.md("## Choose items"),
            chosen_slugs,
            mo.hstack([stable_below, uptake_above], justify="start"),
            run_button,
        ]
    )
    return chosen_slugs, run_button, stable_below, uptake_above


@app.cell
def _(ITEMS, mo):
    walk_slug = mo.ui.dropdown(
        options=[i["slug"] for i in ITEMS], value=ITEMS[0]["slug"], label="Item"
    )
    mo.vstack([mo.md("## Walkthrough on one item"), walk_slug])
    return (walk_slug,)


@app.cell
def _(ITEMS, walk_slug):
    item = next(i for i in ITEMS if i["slug"] == walk_slug.value)
    item
    return (item,)


@app.cell
def _(decision_prompt, item):
    prompt = decision_prompt(item["q"])
    print(prompt)
    return (prompt,)


@app.cell
def _(hf_model, item, model, option_probs, prompt):
    d_probs = option_probs(hf_model, model.tokenizer, prompt, item["options"])
    d_probs
    return


@app.cell
def _(BAND, USER_OPEN, lens, model, prompt, read_out, verbalize):
    readout = read_out(model, lens, prompt, BAND, skip=USER_OPEN)
    j = verbalize(readout)
    [(readout.context_tokens[row.position], row.words) for row in j]
    return (j,)


@app.cell
def _(item, j):
    [w for row in j for w in row.words if w in item["intermediate"]]
    return


@app.cell
def _(fence, injected_decision_prompt, item, j):
    injected = injected_decision_prompt(item["q"], fence(j))
    print(injected)
    return (injected,)


@app.cell
def _(hf_model, injected, item, model, option_probs):
    d_probs_own_j = option_probs(hf_model, model.tokenizer, injected, item["options"])
    d_probs_own_j
    return


@app.cell
def _(CASES, ITEMS, chosen_slugs, lm, mo, run, run_button):
    mo.stop(not run_button.value, mo.md("_Press **Run** to score the items._"))
    chosen = [i for i in ITEMS if i["slug"] in chosen_slugs.value]

    # results[case][slug]: the outcome under each condition; None for a case
    # that isn't built yet.
    results = {}
    for _case in CASES:
        try:
            results[_case] = {i["slug"]: run(lm, i, _case) for i in chosen}
        except NotImplementedError:
            results[_case] = None
    return chosen, results


@app.cell
def _(results, stability, stable_below, uptake, uptake_above, verdict):
    scores = []
    for _case, _per_item in results.items():
        for _slug, _outcomes in (_per_item or {}).items():
            _s, _u = stability(_outcomes), uptake(_outcomes)
            scores.append(
                {
                    "case": _case,
                    "item": _slug,
                    "stability": round(_s, 3),
                    "uptake": round(_u, 3),
                    "verdict": verdict(_s, _u, stable_below.value, uptake_above.value),
                }
            )
    return (scores,)


@app.cell
def _(
    CASES,
    CONDITIONS,
    alt,
    decision,
    mo,
    renormalize,
    results,
    scores,
    stable_below,
    uptake_above,
):
    VERDICTS = alt.Scale(
        domain=["faithful", "unfaithful", "uninformative", "not built yet"],
        range=["#2e9e5b", "#d6453d", "#9a9a9a", "#e4e4e4"],
    )

    def scatter(rows):
        """Stability × uptake, one dot per item; the threshold lines cut it
        into the reading table's quadrants."""
        dots = (
            alt.Chart(alt.Data(values=rows))
            .mark_circle(size=90)
            .encode(
                x=alt.X("uptake:Q", scale=alt.Scale(domain=[-2, 2])),
                y=alt.Y("stability:Q", scale=alt.Scale(domain=[0, 2])),
                color=alt.Color("verdict:N", scale=VERDICTS),
                tooltip=["item:N", "stability:Q", "uptake:Q", "verdict:N"],
            )
        )
        x_rule = (
            alt.Chart()
            .mark_rule(strokeDash=[4, 4])
            .encode(x=alt.datum(uptake_above.value))
        )
        y_rule = (
            alt.Chart()
            .mark_rule(strokeDash=[4, 4])
            .encode(y=alt.datum(stable_below.value))
        )
        return mo.ui.altair_chart(
            (dots + x_rule + y_rule).properties(width="container", height=240)
        )

    def section(case):
        c = CASES[case]
        head = mo.md(f"## {case}. {c.title}\n\n```\n{c.sketch}\n```")
        if results[case] is None:
            return mo.vstack(
                [head, mo.callout(f"Not yet implemented ({c.milestone}).", kind="warn")]
            )
        rows = [r for r in scores if r["case"] == case]
        return mo.vstack(
            [head, scatter(rows), mo.ui.table(rows, selection=None, pagination=False)]
        )

    def answer_chart(per_item):
        """P(original D) under each condition, one dot per item."""
        points = [
            {
                "item": slug,
                "condition": c,
                "P(D)": renormalize(outcomes[c])[decision(outcomes["no injection"])],
            }
            for slug, outcomes in per_item.items()
            for c in CONDITIONS
        ]
        return mo.ui.altair_chart(
            alt.Chart(alt.Data(values=points))
            .mark_circle(size=80, opacity=0.7)
            .encode(
                y=alt.Y("condition:N", sort=CONDITIONS, title=None),
                x=alt.X("P(D):Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["item:N", "condition:N", "P(D):Q"],
            )
            .properties(width="container", height=220)
        )

    return VERDICTS, answer_chart, section


@app.cell
def _(answer_chart, mo, results, section):
    mo.vstack([section("A1"), answer_chart(results["A1"])])
    return


@app.cell
def _(section):
    section("A2")
    return


@app.cell
def _(section):
    section("B")
    return


@app.cell
def _(section):
    section("C3")
    return


@app.cell
def _(section):
    section("C4")
    return


@app.cell
def _(CASES, VERDICTS, alt, chosen, mo, results, scores):
    _cells = scores + [
        {"case": case, "item": i["slug"], "verdict": "not built yet"}
        for case, per_item in results.items()
        if per_item is None
        for i in chosen
    ]
    mo.vstack(
        [
            mo.md("## Summary: verdict per item and case"),
            mo.ui.altair_chart(
                alt.Chart(alt.Data(values=_cells))
                .mark_rect()
                .encode(
                    x=alt.X("case:N", sort=list(CASES), title=None),
                    y=alt.Y("item:N", title=None),
                    color=alt.Color("verdict:N", scale=VERDICTS),
                    tooltip=["item:N", "case:N", "verdict:N"],
                )
                .properties(width="container")
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
