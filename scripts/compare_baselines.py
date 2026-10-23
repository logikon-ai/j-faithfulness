import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import gzip
    import json

    import marimo as mo
    import pandas as pd

    from j_faithfulness.items import daily_dilemmas_rows, daily_dilemmas_values
    from j_faithfulness.lens import LatentReadout
    from j_faithfulness.verbalize import (
        echoes,
        fence,
        verbalize,
        verbalize_against,
        words,
    )

    return (
        LatentReadout,
        daily_dilemmas_rows,
        daily_dilemmas_values,
        echoes,
        fence,
        gzip,
        json,
        mo,
        pd,
        verbalize,
        verbalize_against,
        words,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Compare baselines and contrasts

    Builds J six ways from the read-outs written by `read_baselines.py`: the
    typist, the copy editor or both as baseline, each by subtraction or by
    the graded contrast. Runs locally; no GPU.
    """)
    return


@app.cell
def _(LatentReadout, daily_dilemmas_rows, daily_dilemmas_values, gzip, json, mo):
    READOUTS = mo.notebook_dir() / "baseline-readouts.json.gz"

    with gzip.open(READOUTS, "rt", encoding="utf-8") as f:
        stored = json.load(f)
    # (id, protocol, kind) → role → read-out
    transcripts = {}
    for entry in stored:
        key = (entry["id"], entry["protocol"], entry["kind"])
        transcripts.setdefault(key, {})[entry["role"]] = LatentReadout(
            **entry["readout"]
        )
    values = daily_dilemmas_values(daily_dilemmas_rows())

    print(f"{len(transcripts)} transcripts from {READOUTS.name}")
    return transcripts, values


@app.cell
def _(mo):
    min_score = mo.ui.slider(
        0.0, 2.0, step=0.1, value=1.0, label="Graded contrast: minimum score"
    )
    min_score
    return (min_score,)


@app.cell
def _(min_score, verbalize, verbalize_against):
    def subtracted(*baselines):
        return lambda r: verbalize_against(r["advisor"], *(r[b] for b in baselines))

    def graded(*baselines):
        # The subtraction's loose settings, so only the contrast differs.
        return lambda r: verbalize(
            r["advisor"],
            [r[b] for b in baselines],
            k=10,
            min_tf=0.5,
            min_score=min_score.value,
        )

    variants = {
        "subtract typist": subtracted("typist"),
        "subtract editor": subtracted("editor"),
        "subtract both": subtracted("typist", "editor"),
        "graded typist": graded("typist"),
        "graded editor": graded("editor"),
        "graded both": graded("typist", "editor"),
    }
    return (variants,)


@app.cell
def _(transcripts, variants):
    js = {
        (key, name): build(roles)
        for key, roles in transcripts.items()
        for name, build in variants.items()
    }
    return (js,)


@app.cell
def _(echoes, js, mo, pd, values, words):
    # Process words the model readies to answer or rework a text with, not
    # considerations of the dilemma.
    META = {
        "comment",
        "commentary",
        "rationale",
        "summary",
        "brief",
        "briefing",
        "text",
        "paraphrase",
        "answer",
        "response",
        "reply",
        "explanation",
        "analysis",
    }
    # Value labels like "respect for others": their content words.
    LABEL_FILLERS = {"for", "of", "and", "the", "to", "in", "self", "others"}

    def value_words(item_id):
        return {
            w
            for value in values[item_id]
            for w in words(value)
            if w not in LABEL_FILLERS
        }

    def measures(item_id, j):
        j_words = [w for row in j for w in row.words]
        on_topic = [w for w in j_words if echoes(w, value_words(item_id))]
        return {
            "rows": len(j),
            "words": len(j_words),
            "on topic": len(on_topic) / len(j_words) if j_words else 0.0,
            "meta words": sum(echoes(w, META) for w in j_words),
        }

    scored = pd.DataFrame(
        [
            {
                "id": item_id,
                "case": f"{kind}, protocol {protocol}",
                "variant": name,
                **measures(item_id, j),
            }
            for ((item_id, protocol, kind), name), j in js.items()
        ]
    )

    mo.ui.table(
        scored.groupby(["case", "variant"], sort=False)[
            ["rows", "words", "on topic", "meta words"]
        ]
        .mean()
        .round(2)
        .reset_index(),
        label="Mean per transcript, by case and variant:",
    )
    return


@app.cell
def _(fence, js, transcripts, values, variants):
    ITEM_IDX = 0
    CASE = (4, "explanation")

    item_id = list(dict.fromkeys(key[0] for key in transcripts))[ITEM_IDX]
    print(f"J six ways for {item_id}, {CASE[1]}, protocol {CASE[0]}.")
    print(f"Annotated values: {', '.join(values[item_id])}\n")
    for name in variants:
        print(f"[[{name}]]\n{fence(js[((item_id, *CASE), name)])}\n")
    return


if __name__ == "__main__":
    app.run()
