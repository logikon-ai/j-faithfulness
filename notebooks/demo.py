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
    import random

    import altair as alt
    import marimo as mo

    from j_faithfulness.ablations import (
        distinct_words,
        drop_random,
        drop_top,
        present,
        swap,
        swap_random,
    )
    from j_faithfulness.compare import brier, renormalize
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
        ITEMS,
        USER_OPEN,
        alt,
        brier,
        decision,
        decision_prompt,
        distinct_words,
        drop_random,
        drop_top,
        fence,
        injected_decision_prompt,
        load,
        mo,
        option_probs,
        present,
        random,
        read_out,
        renormalize,
        swap,
        swap_random,
        verbalize,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Case A in protocol 1: does D respond to its latent reasoning?

    For each one-hop decision (guacamole → avocado → B), read J over Q and the
    generation prompt, insert it at the end of the user turn with the notice,
    and see how far the answer moves from the original D (Brier score against
    D: 0 = all mass on D, 2 = all on the other option). The contrasts that
    count (DESIGN.md):

    - **Stability:** own J and the notice alone stay near no J.
    - **Sensitivity:** swapping the intermediate for a foil that implies the
      other answer (avocado → chickpea) moves D more than putting the same
      foil in place of a random word; dropping the most salient words moves
      it more than dropping random ones.
    """)
    return


@app.cell
def _(load):
    hf_model, model, lens = load()
    return hf_model, lens, model


@app.cell
def _():
    N_SEEDS = 5
    DROP_N = 3
    CONDITIONS = [
        "no J",
        "notice only",
        "own J",
        "intermediate → foil",
        "random word → foil",
        f"drop top {DROP_N}",
        f"drop random {DROP_N}",
    ]
    return CONDITIONS, DROP_N, N_SEEDS


@app.cell
def _(
    BAND,
    DROP_N,
    ITEMS,
    N_SEEDS,
    USER_OPEN,
    brier,
    decision,
    decision_prompt,
    distinct_words,
    drop_random,
    drop_top,
    fence,
    hf_model,
    injected_decision_prompt,
    lens,
    model,
    option_probs,
    present,
    random,
    read_out,
    renormalize,
    swap,
    swap_random,
    verbalize,
):
    def probs_with(item, j):
        """Answer probabilities with J injected (None: notice only)."""
        prompt = injected_decision_prompt(item["q"], None if j is None else fence(j))
        return option_probs(hf_model, model.tokenizer, prompt, item["options"])

    def outcome(probs_list, d):
        """Mean over runs: Brier against d, P(d) and the mass on the options."""
        n = len(probs_list)
        return {
            "brier": sum(brier(p, d) for p in probs_list) / n,
            "P(D)": sum(renormalize(p)[d] for p in probs_list) / n,
            "mass": sum(sum(p.values()) for p in probs_list) / n,
        }

    runs = {}
    for _item in ITEMS:
        _prompt = decision_prompt(_item["q"])
        _p0 = option_probs(hf_model, model.tokenizer, _prompt, _item["options"])
        _d = decision(_p0)
        _j = verbalize(read_out(model, lens, _prompt, band=BAND, skip=USER_OPEN))
        _found = present(_j, _item["foils"])
        _top = distinct_words(_j)[:DROP_N]
        _rngs = [random.Random(seed) for seed in range(N_SEEDS)]
        _cond = {
            "no J": [_p0],
            "notice only": [probs_with(_item, None)],
            "own J": [probs_with(_item, _j)],
            f"drop top {DROP_N}": [probs_with(_item, drop_top(_j, DROP_N))],
            f"drop random {DROP_N}": [
                probs_with(_item, drop_random(_j, DROP_N, rng, exclude=_top))
                for rng in _rngs
            ],
        }
        if _found:
            _cond["intermediate → foil"] = [probs_with(_item, swap(_j, _item["foils"]))]
            _cond["random word → foil"] = [
                probs_with(
                    _item,
                    swap_random(_j, _item["foils"], rng, exclude=_item["intermediate"]),
                )
                for rng in [random.Random(seed) for seed in range(N_SEEDS)]
            ]
        runs[_item["slug"]] = {
            "D": _d,
            "J": _j,
            "found": _found,
            "outcomes": {c: outcome(ps, _d) for c, ps in _cond.items()},
        }
    return (runs,)


@app.cell
def _(CONDITIONS, ITEMS, runs):
    def _cell(slug, condition):
        o = runs[slug]["outcomes"].get(condition)
        return "" if o is None else round(o["brier"], 3)

    contrast = [
        {
            "item": i["slug"],
            "D": runs[i["slug"]]["D"],
            "intermediate in J": ", ".join(runs[i["slug"]]["found"]),
            **{c: _cell(i["slug"], c) for c in CONDITIONS},
        }
        for i in ITEMS
    ]
    points = [
        {
            "item": slug,
            "condition": c,
            "brier": o["brier"],
            "P(D)": o["P(D)"],
            "mass on options": o["mass"],
        }
        for slug, run in runs.items()
        for c, o in run["outcomes"].items()
    ]
    return contrast, points


@app.cell
def _(CONDITIONS, alt, mo, points):
    _base = alt.Chart(alt.Data(values=points)).encode(
        y=alt.Y("condition:N", sort=CONDITIONS, title=None),
        x=alt.X(
            "brier:Q",
            scale=alt.Scale(domain=[0, 2]),
            title="Brier score against the original D (0 = unchanged)",
        ),
    )
    _dots = _base.mark_circle(size=80, opacity=0.6, color="#2a78d6").encode(
        tooltip=[
            "item:N",
            "condition:N",
            alt.Tooltip("brier:Q", format=".3f"),
            alt.Tooltip("P(D):Q", format=".3f"),
            alt.Tooltip("mass on options:Q", format=".3f"),
        ]
    )
    _mean = _base.mark_tick(thickness=2, size=22, color="#2a78d6").encode(
        x="mean(brier):Q", tooltip=[alt.Tooltip("mean(brier):Q", format=".3f")]
    )
    mo.vstack(
        [
            mo.md(
                "## How far each condition moves D\n\nOne dot per item, the tick "
                "is the mean. Random conditions are averaged over seeds."
            ),
            mo.ui.altair_chart(
                (_dots + _mean).properties(width="container", height=260)
            ),
        ]
    )
    return


@app.cell
def _(contrast, mo):
    mo.vstack(
        [
            mo.md("## Contrast table (Brier against D)"),
            mo.ui.table(contrast, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(ITEMS, mo):
    item_select = mo.ui.dropdown(
        options=[i["slug"] for i in ITEMS], value=ITEMS[0]["slug"], label="Item"
    )
    item_select
    return (item_select,)


@app.cell
def _(
    CONDITIONS,
    ITEMS,
    fence,
    injected_decision_prompt,
    item_select,
    mo,
    runs,
    swap,
):
    _item = next(i for i in ITEMS if i["slug"] == item_select.value)
    _run = runs[_item["slug"]]
    _rows = [
        {"condition": c, **{k: round(v, 3) for k, v in _run["outcomes"][c].items()}}
        for c in CONDITIONS
        if c in _run["outcomes"]
    ]
    mo.vstack(
        [
            mo.md(f"**Q:** {_item['q']}\n\n**D:** {_run['D']}"),
            mo.ui.table(_rows, selection=None, pagination=False),
            mo.md("**Own J, as injected:**"),
            mo.plain_text(injected_decision_prompt(_item["q"], fence(_run["J"]))),
            mo.md("**Intermediate → foil:**"),
            mo.plain_text(fence(swap(_run["J"], _item["foils"])))
            if _run["found"]
            else mo.md("_The intermediate isn't in J._"),
        ]
    )
    return


@app.cell
def _(ITEMS, contrast, fence, runs):
    # Plain-text dump: an .ipynb export keeps printed text, but renders
    # tables and charts as images.
    for _row in contrast:
        print(_row)
    for _item in ITEMS:
        _run = runs[_item["slug"]]
        print(f"\n-- {_item['slug']}  D={_run['D']}")
        for _c, _o in _run["outcomes"].items():
            print(f"{_c:>22}: " + ", ".join(f"{k} {v:.3f}" for k, v in _o.items()))
        print(fence(_run["J"]))
    return


if __name__ == "__main__":
    app.run()
