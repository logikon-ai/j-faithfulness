# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
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
    import marimo as mo

    from j_faithfulness.items import load_daily_dilemmas, split
    from j_faithfulness.lens import BAND, load, read_out
    from j_faithfulness.protocols import (
        READOUT_TASKS,
        USER_OPEN,
        decision,
        decision_prompt,
        option_probs,
        task_prompt,
    )
    from j_faithfulness.verbalize import in_context, subtract, touches, verbalize

    return (
        BAND,
        READOUT_TASKS,
        USER_OPEN,
        decision,
        decision_prompt,
        in_context,
        load,
        load_daily_dilemmas,
        mo,
        option_probs,
        read_out,
        split,
        subtract,
        task_prompt,
        touches,
        verbalize,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Contrastive read-out (PLAN.md, M1)

    Thinking off. Does J surface the annotated values of each option, and of
    D's option in particular? J as built now (TF × IDF over Q alone) against
    J scored by the read-out of Q under the *decide* task, relative to the
    read-out under *copy* or *plain English* (difference or log ratio of TF).
    Values that occur in Q can't show in J (context echo is dropped), so they
    are counted apart.
    """)
    return


@app.cell
def _(load):
    lm = load()
    hf_model, model, lens = lm
    return hf_model, lens, model


@app.cell
def _(load_daily_dilemmas, split):
    items = split(load_daily_dilemmas(), "dev", 10)
    [(i.id, i.domain) for i in items]
    return (items,)


@app.cell
def _(READOUT_TASKS):
    READOUT_TASKS
    return


@app.cell
def _(
    BAND,
    USER_OPEN,
    decision,
    decision_prompt,
    hf_model,
    items,
    lens,
    model,
    option_probs,
    read_out,
    task_prompt,
):
    # Per item: D, and the read-out of Q without a task and under each task.
    d = {}
    readouts = {}
    for _item in items:
        _prompt = decision_prompt(_item.q)
        d[_item.id] = decision(
            option_probs(hf_model, model.tokenizer, _prompt, _item.options)
        )
        readouts[_item.id] = {
            "none": read_out(model, lens, _prompt, BAND, skip=USER_OPEN)
        }
        for _task in ["copy", "plain", "decide"]:
            _p, _skip = task_prompt(_item.q, _task)
            readouts[_item.id][_task] = read_out(model, lens, _p, BAND, skip=_skip)
    d
    return d, readouts


@app.cell
def _(items, readouts, subtract, verbalize):
    # Variant → (read-out, baseline, measure).
    VARIANTS = {
        "TF×IDF": ("none", None, "diff"),
        "TF×IDF · decide": ("decide", None, "diff"),
        "decide − copy": ("decide", "copy", "diff"),
        "decide − plain": ("decide", "plain", "diff"),
        "decide / copy": ("decide", "copy", "ratio"),
        "decide / plain": ("decide", "plain", "ratio"),
    }
    js = {
        v: {
            i.id: verbalize(
                readouts[i.id][r],
                readouts[i.id][b] if b else None,
                measure,
            )
            for i in items
        }
        for v, (r, b, measure) in VARIANTS.items()
    }

    # The decide J minus every word of a baseline's J, both built loose
    # (k = 50, TF ≥ 0.5); then the top 10 per position.
    def _loose(readout):
        return verbalize(readout, k=50, min_tf=0.5)

    for _b in ["copy", "plain"]:
        VARIANTS[f"decide ∖ {_b}"] = ("decide", _b, "subtract")
        js[f"decide ∖ {_b}"] = {
            i.id: subtract(
                _loose(readouts[i.id]["decide"]), _loose(readouts[i.id][_b]), k=10
            )
            for i in items
        }
    return VARIANTS, js


@app.cell
def _(d, in_context, items, js, touches):
    # Per variant and item: values of D's option and of the other option that
    # J touches, out of those not in Q; values in Q; the number of J's words.
    rows = []
    for _v, _per_item in js.items():
        for _item in items:
            _j = _per_item[_item.id]
            _row = {"variant": _v, "item": _item.id}
            for _side, _labels in [
                ("D", [d[_item.id]]),
                ("other", [o for o in _item.options if o != d[_item.id]]),
            ]:
                _values = [c for o in _labels for c in _item.concepts[o]]
                _free = [c for c in _values if not in_context(c, _item.q)]
                _row[f"{_side} hit"] = sum(touches(_j, c) for c in _free)
                _row[f"{_side} free"] = len(_free)
                _row[f"{_side} in Q"] = len(_values) - len(_free)
            _row["J words"] = sum(len(r.words) for r in _j)
            rows.append(_row)
    rows
    return (rows,)


@app.cell
def _(VARIANTS, rows):
    # Totals over the items.
    _keys = ["D hit", "D free", "other hit", "other free", "D in Q", "other in Q"]
    totals = [
        {"variant": v}
        | {k: sum(r[k] for r in rows if r["variant"] == v) for k in _keys + ["J words"]}
        for v in VARIANTS
    ]
    totals
    return


@app.cell
def _(items, mo):
    item_id = mo.ui.dropdown(
        options=[i.id for i in items], value=items[0].id, label="Item"
    )
    item_id
    return (item_id,)


@app.cell
def _(d, item_id, items):
    item = next(i for i in items if i.id == item_id.value)
    print(item.q)
    print("D =", d[item.id])
    item.concepts
    return (item,)


@app.cell
def _(VARIANTS, item, js, readouts):
    # Each variant's J, row by row, at the word it comes from: look for topic
    # and format words.
    {
        v: [
            (
                readouts[item.id][VARIANTS[v][0]].context_tokens[row.position],
                ", ".join(row.words),
            )
            for row in per_item[item.id]
        ]
        for v, per_item in js.items()
    }
    return


if __name__ == "__main__":
    app.run()
