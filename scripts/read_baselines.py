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
    import dataclasses
    import gzip
    import json

    import datasets
    import marimo as mo

    from j_faithfulness.formats import FORMATS
    from j_faithfulness.lens import load, read_span

    return FORMATS, dataclasses, datasets, gzip, json, load, mo, read_span


@app.cell
def _(mo):
    mo.md(r"""
    # Read out J's spans under the advisor and the baselines

    Reads J's span of the first dev items under three roles: the advisor, the
    typist and the copy editor, for reasoning and for explanation in
    protocols 3 and 4. Writes the read-outs to one file, so
    `compare_baselines.py` can build J every way locally, without the GPU.
    Download the file when the run is done.
    """)
    return


@app.cell
def _(datasets):
    MODEL_ID = "Qwen/Qwen3.5-4B"
    DATASET = "ggbetz/cached-reasoning-records"
    SHA = "dc44f7efbee7a35b04ccf25b947d6145030088b1"
    N_ITEMS = 20
    OUT = "baseline-readouts.json.gz"

    records = datasets.load_dataset(DATASET, revision=SHA, split="train")
    rows = [row for row in records if row["metadata"]["model_id"] == MODEL_ID]
    item_ids = list(dict.fromkeys(row["id"] for row in rows))[:N_ITEMS]
    cases = [
        (row, kind)
        for row in rows
        if row["id"] in item_ids
        for kind in (("reasoning", "explanation") if row["r"] else ("explanation",))
    ]
    print(f"{len(item_ids)} items, {len(cases)} cases, 3 roles each")
    return MODEL_ID, OUT, cases


@app.cell
def _(FORMATS, MODEL_ID, load):
    fmt = FORMATS[MODEL_ID]
    model, lens = load(MODEL_ID)
    roles = {"advisor": None, "typist": fmt.COPY_ROLE, "editor": fmt.EDIT_ROLE}
    return fmt, lens, model, roles


@app.cell
def _(OUT, cases, dataclasses, fmt, gzip, json, lens, mo, model, read_span, roles):
    readouts = []
    with mo.status.progress_bar(total=len(cases), title="Read out") as bar:
        for row, kind in cases:
            protocol = 3 if row["r"] is None else 4
            bar.update(
                increment=0, subtitle=f"{row['id']}: {kind}, protocol {protocol}"
            )
            for name, role in roles.items():
                readout = read_span(model, lens, fmt, kind, row["q"], row["r"], role)
                readouts.append(
                    {
                        "id": row["id"],
                        "protocol": protocol,
                        "kind": kind,
                        "role": name,
                        "readout": dataclasses.asdict(readout),
                    }
                )
            bar.update()
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        json.dump(readouts, f)

    print(f"Wrote {len(readouts)} read-outs to {OUT}.")
    return


if __name__ == "__main__":
    app.run()
