# Local, no GPU: tune J on a read-out dump downloaded from notebooks/readout.py.
# Run from the repo with `uv run marimo edit notebooks/tune.py`.

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import gzip
    import json

    import marimo as mo

    from j_faithfulness.lens import BAND, readouts_from_json
    from j_faithfulness.verbalize import fence, verbalize

    return BAND, fence, gzip, json, mo, readouts_from_json, verbalize


@app.cell
def _(mo):
    mo.md(r"""
    # Tuning J offline

    Loads the read-outs that `notebooks/readout.py` downloads (all layers, top
    100 per ranked list) and verbalizes them with the settings below.
    """)
    return


@app.cell
def _(mo):
    path = mo.ui.text(value="readouts.json.gz", label="Dump", full_width=True)
    path
    return (path,)


@app.cell
def _(gzip, json, path, readouts_from_json):
    with gzip.open(path.value) as _f:
        _data = json.load(_f)
    ITEMS = _data["items"]
    probs = _data["probs"]
    readouts = readouts_from_json(_data)
    return ITEMS, probs, readouts


@app.cell
def _(BAND, mo):
    band = mo.ui.range_slider(start=0, stop=30, value=list(BAND), label="Band")
    k = mo.ui.slider(1, 10, value=5, label="k")
    min_tf = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="min_tf")
    min_score = mo.ui.slider(0.0, 5.0, step=0.1, value=1.0, label="min_score")
    min_zipf = mo.ui.slider(0.0, 5.0, step=0.1, value=2.5, label="min_zipf")
    mo.hstack([band, k, min_tf, min_score, min_zipf], justify="start")
    return band, k, min_score, min_tf, min_zipf


@app.cell
def _(ITEMS, band, k, min_score, min_tf, min_zipf, readouts, verbalize):
    js = {
        i["slug"]: verbalize(
            readouts[i["slug"]].in_band(*band.value),
            k=k.value,
            min_tf=min_tf.value,
            min_score=min_score.value,
            min_zipf=min_zipf.value,
        )
        for i in ITEMS
    }
    return (js,)


@app.cell
def _(ITEMS, js, mo, probs):
    def hit(item):
        found = {w.casefold() for row in js[item["slug"]] for w in row.words}
        return any(t.casefold() in found for t in item["intermediate"])

    mo.ui.table(
        [
            {
                "item": i["slug"],
                "P(options)": str(probs[i["slug"]]),
                "rows": len(js[i["slug"]]),
                "words": sum(len(row.words) for row in js[i["slug"]]),
                "intermediate in J": hit(i) if i["intermediate"] else "",
            }
            for i in ITEMS
        ],
        selection=None,
        pagination=False,
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
def _(fence, item_select, js, mo, readouts):
    _r = readouts[item_select.value]
    _j = js[item_select.value]
    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "token": repr(_r.context_tokens[row.position]),
                        "words": ", ".join(row.words),
                    }
                    for row in _j
                ],
                selection=None,
                pagination=False,
            ),
            mo.plain_text(fence(_j)),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
