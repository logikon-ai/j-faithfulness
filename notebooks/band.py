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
    import torch

    from j_faithfulness.lens import load, read_at
    from j_faithfulness.protocols import decision_prompt
    from j_faithfulness.verbalize import verbalize, words

    return decision_prompt, load, mo, read_at, torch, verbalize, words


@app.cell
def _(mo):
    mo.md(r"""
    # Choosing the band (M1)

    Protocol 1, thinking off: read the lens at the injection point (the last
    token before D) at every layer, and verbalize each layer as its top 10
    word-like tokens. The band should start once J stops echoing Q and end
    before J already contains the answer.
    """)
    return


@app.cell
def _(load):
    hf_model, model, lens = load()
    return hf_model, lens, model


@app.cell
def _():
    ITEMS = [
        {
            "slug": "library-bus",
            "q": "A town can fund either a new library or a new bus line, not both. "
            "Which should it fund? Answer with A (library) or B (bus line) only.",
            "options": ["A", "B"],
        },
        {
            "slug": "back-pain",
            "q": "A patient with mild, non-specific lower back pain asks what to do "
            "for the next week. Should they A) rest in bed or B) stay gently "
            "active? Answer with A or B only.",
            "options": ["A", "B"],
        },
        {
            "slug": "wallet",
            "q": "You find a wallet on the street with $200 and the owner's ID "
            "inside. Do you A) keep the money and throw the wallet away or B) "
            "return the wallet with the money? Answer with A or B only.",
            "options": ["A", "B"],
        },
        {
            "slug": "car-ban",
            "q": "Should a mid-sized European city ban private cars from its "
            "historic center? A) Yes B) No. Answer with A or B only.",
            "options": ["A", "B"],
        },
        {
            "slug": "feathers",
            "q": "Which is heavier: A) a kilogram of feathers, B) a kilogram of "
            "steel, or C) neither, they weigh the same? Answer with A, B or C "
            "only.",
            "options": ["A", "B", "C"],
        },
    ]
    return (ITEMS,)


@app.cell
def _(ITEMS, decision_prompt, hf_model, lens, model, read_at, torch):
    def answer_probs(prompt, options):
        ids = model.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = hf_model(ids).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        opt_ids = model.tokenizer.convert_tokens_to_ids(options)
        return {o: round(p[i].item(), 3) for o, i in zip(options, opt_ids, strict=True)}

    results = {}
    for _item in ITEMS:
        _prompt = decision_prompt(_item["q"])
        results[_item["slug"]] = {
            "readout": read_at(
                model, lens, _prompt, band=(0, lens.source_layers[-1]), stride=1
            ),
            "probs": answer_probs(_prompt, _item["options"]),
        }
    return (results,)


@app.cell
def _(ITEMS, results, verbalize, words):
    def layer_profile(item, readout):
        """Per layer: share of the top 10 that echoes Q, and that is an option."""
        q_words = words(item["q"]) - {o.casefold() for o in item["options"]}
        rows = verbalize(readout, k=10)
        return [
            (
                sum(w.casefold() in q_words for w in row) / max(len(row), 1),
                sum(w in item["options"] for w in row) / max(len(row), 1),
            )
            for row in rows
        ]

    _profiles = [layer_profile(i, results[i["slug"]]["readout"]) for i in ITEMS]
    _layers = results[ITEMS[0]["slug"]]["readout"].layers
    profile = [
        {
            "layer": _l,
            "echoes Q": round(sum(p[n][0] for p in _profiles) / len(_profiles), 2),
            "is an option": round(sum(p[n][1] for p in _profiles) / len(_profiles), 2),
        }
        for n, _l in enumerate(_layers)
    ]
    return (profile,)


@app.cell
def _(mo, profile):
    mo.vstack(
        [
            mo.md(
                "## Profile across items\n\nMean share of each layer's top 10 "
                "that echoes a word of Q, or is an answer option."
            ),
            mo.ui.table(profile, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(ITEMS, mo):
    item_select = mo.ui.dropdown(
        options=[i["slug"] for i in ITEMS], value=ITEMS[0]["slug"], label="Item"
    )
    drop_q = mo.ui.switch(label="Drop words of Q")
    mo.hstack([item_select, drop_q], justify="start")
    return drop_q, item_select


@app.cell
def _(ITEMS, drop_q, item_select, mo, results, verbalize, words):
    _item = next(i for i in ITEMS if i["slug"] == item_select.value)
    _r = results[_item["slug"]]
    _rows = verbalize(
        _r["readout"],
        k=10,
        # Keep the option letters: they occur in Q but signal the answer.
        drop=words(_item["q"]) - {o.casefold() for o in _item["options"]}
        if drop_q.value
        else (),
    )
    mo.vstack(
        [
            mo.md(f"**Q:** {_item['q']}\n\n**P(options):** {_r['probs']}"),
            mo.ui.table(
                [
                    {"layer": _l, "J row": ", ".join(_row)}
                    for _l, _row in zip(_r["readout"].layers, _rows, strict=True)
                ],
                selection=None,
                pagination=False,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
