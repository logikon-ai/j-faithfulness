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

    from j_faithfulness.lens import load, read_out
    from j_faithfulness.protocols import USER_OPEN, decision_prompt
    from j_faithfulness.verbalize import fence, normalize, verbalize

    return (
        USER_OPEN,
        decision_prompt,
        fence,
        load,
        mo,
        normalize,
        read_out,
        torch,
        verbalize,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Validating the read-out (M1)

    Protocol 1, thinking off, on decisions that need one hidden step. Each
    item has a known intermediate that is not in Q (guacamole → avocado).
    J has one row per word of the context: its words by TF × IDF over the
    layers of the band, where each ranked list (one position, one layer) is a
    document and IDF counts the documents at the other positions. The
    read-out works if J surfaces the intermediate, and the band should be
    where it does, before the answer takes over.
    `library-bus` is a control without an intermediate.
    """)
    return


@app.cell
def _(load):
    hf_model, model, lens = load()
    return hf_model, lens, model


@app.cell
def _():
    _AB = "Answer with A or B only."
    ITEMS = [
        {
            "slug": "guacamole",
            "q": "My friend is allergic to the main ingredient of guacamole. "
            f"Should I bring A) guacamole or B) hummus to her party? {_AB}",
            "options": ["A", "B"],
            "intermediate": ["avocado"],
        },
        {
            "slug": "boot",
            "q": "Does the country shaped like a boot use the euro as its "
            f"currency? A) yes B) no. {_AB}",
            "options": ["A", "B"],
            "intermediate": ["Italy", "Italian", "Rome"],
        },
        {
            "slug": "eiffel",
            "q": "I'm travelling to the city with the Eiffel Tower. Which "
            f"phrasebook should I pack: A) German or B) French? {_AB}",
            "options": ["A", "B"],
            "intermediate": ["Paris", "France"],
        },
        {
            "slug": "hamlet",
            "q": "Was the author of Hamlet born before the year 1600? A) yes "
            f"B) no. {_AB}",
            "options": ["A", "B"],
            "intermediate": ["Shakespeare"],
        },
        {
            "slug": "kangaroo",
            "q": "I want to see kangaroos in the wild. Should I book a flight "
            f"to A) Sydney or B) Nairobi? {_AB}",
            "options": ["A", "B"],
            "intermediate": ["Australia"],
        },
        {
            "slug": "honey",
            "q": "My son is allergic to stings from the insect that makes "
            "honey. Should he join A) a visit to an apiary or B) a museum "
            f"tour? {_AB}",
            "options": ["A", "B"],
            "intermediate": ["bee", "bees"],
        },
        {
            "slug": "red-planet",
            "q": f"Is the Red Planet larger than Earth? A) yes B) no. {_AB}",
            "options": ["A", "B"],
            "intermediate": ["Mars"],
        },
        {
            "slug": "ocean",
            "q": f"Does the largest ocean on Earth border Japan? A) yes B) no. {_AB}",
            "options": ["A", "B"],
            "intermediate": ["Pacific"],
        },
        {
            "slug": "library-bus",
            "q": "A town can fund either a new library or a new bus line, not "
            f"both. Which should it fund? A) library B) bus line. {_AB}",
            "options": ["A", "B"],
            "intermediate": [],
        },
    ]
    return (ITEMS,)


@app.cell
def _(ITEMS, USER_OPEN, decision_prompt, hf_model, lens, model, read_out, torch):
    def answer_probs(prompt, options):
        ids = model.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = hf_model(ids).logits[0, -1].float()
        p = torch.softmax(logits, -1)
        opt_ids = model.tokenizer.convert_tokens_to_ids(options)
        return {o: round(p[i].item(), 3) for o, i in zip(options, opt_ids, strict=True)}

    # Every layer the lens covers, at every word end after the opening
    # template; bands are cut from this.
    results = {}
    for _item in ITEMS:
        _prompt = decision_prompt(_item["q"])
        _probs = answer_probs(_prompt, _item["options"])
        results[_item["slug"]] = {
            "readout": read_out(
                model, lens, _prompt, band=(0, lens.source_layers[-1]), skip=USER_OPEN
            ),
            "probs": _probs,
            "D": max(_probs, key=_probs.get),
        }
    return (results,)


@app.cell
def _(ITEMS, normalize, results):
    def best_rank(readout, targets):
        """Per layer: the best raw rank of any target (top 200 only, else None)
        over all positions, and the context token where it occurs."""
        targets = {t.casefold() for t in targets}
        out = []
        for j, _layer in enumerate(readout.layers):
            best = None
            for i, ranked in enumerate(readout.top):
                for rank, token in enumerate(ranked[j]):
                    if normalize(token).casefold() in targets:
                        if best is None or rank < best[0]:
                            best = (rank, readout.context_tokens[readout.positions[i]])
                        break
            out.append(best)
        return out

    def last_rank(readout, target):
        """Per layer: the raw rank of ``target`` at the last position."""
        out = []
        for ranked in readout.top[-1]:
            ranks = [r for r, t in enumerate(ranked) if normalize(t) == target]
            out.append(ranks[0] if ranks else None)
        return out

    _layers = results[ITEMS[0]["slug"]]["readout"].layers
    _with = [i for i in ITEMS if i["intermediate"]]
    _inter = {
        i["slug"]: best_rank(results[i["slug"]]["readout"], i["intermediate"])
        for i in _with
    }
    _answer = {
        i["slug"]: last_rank(results[i["slug"]]["readout"], results[i["slug"]]["D"])
        for i in ITEMS
    }

    def _fmt(best):
        return "" if best is None else f"{best[0]} @{best[1].strip() or repr(best[1])}"

    intermediate_ranks = [
        {"layer": _l, **{s: _fmt(_inter[s][j]) for s in _inter}}
        for j, _l in enumerate(_layers)
    ]
    answer_ranks = [
        {
            "layer": _l,
            **{s: "" if _answer[s][j] is None else _answer[s][j] for s in _answer},
        }
        for j, _l in enumerate(_layers)
    ]
    return answer_ranks, intermediate_ranks


@app.cell
def _(answer_ranks, intermediate_ranks, mo):
    mo.vstack(
        [
            mo.md(
                "## Where the intermediate surfaces\n\nBest raw rank of the "
                "intermediate over all positions, per layer, and the token "
                "where it occurs (blank: not in the top 200)."
            ),
            mo.ui.table(intermediate_ranks, selection=None, pagination=False),
            mo.md(
                "## Where the answer takes over\n\nRaw rank of the answer D at "
                "the last position (blank: not in the top 200)."
            ),
            mo.ui.table(answer_ranks, selection=None, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    band = mo.ui.range_slider(start=0, stop=30, value=[8, 21], label="Band")
    k = mo.ui.slider(1, 10, value=10, label="k")
    min_tf = mo.ui.slider(0.0, 3.0, step=0.1, value=0.5, label="min_tf")
    min_score = mo.ui.slider(0.0, 5.0, step=0.1, value=1.0, label="min_score")
    mo.hstack([band, k, min_tf, min_score], justify="start")
    return band, k, min_score, min_tf


@app.cell
def _(ITEMS, band, k, min_score, min_tf, results, verbalize):
    def j_of(slug, last_only=False):
        r = results[slug]["readout"].in_band(*band.value)
        if last_only:
            r.positions, r.top = r.positions[-1:], r.top[-1:]
        return verbalize(r, k=k.value, min_tf=min_tf.value, min_score=min_score.value)

    def hit(j, item):
        found = {w.casefold() for row in j for w in row.words}
        return any(t.casefold() in found for t in item["intermediate"])

    js = {i["slug"]: j_of(i["slug"]) for i in ITEMS}
    js_last = {i["slug"]: j_of(i["slug"], last_only=True) for i in ITEMS}
    summary = [
        {
            "item": i["slug"],
            "P(options)": str(results[i["slug"]]["probs"]),
            "rows": len(js[i["slug"]]),
            "words": sum(len(row.words) for row in js[i["slug"]]),
            "intermediate in J": hit(js[i["slug"]], i) if i["intermediate"] else "",
            "… in last-position J": hit(js_last[i["slug"]], i)
            if i["intermediate"]
            else "",
        }
        for i in ITEMS
    ]
    return js, js_last, summary


@app.cell
def _(mo, summary):
    mo.vstack(
        [
            mo.md("## J in the band"),
            mo.ui.table(summary, selection=None, pagination=False),
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
def _(fence, item_select, js, mo, results):
    _r = results[item_select.value]["readout"]
    _j = js[item_select.value]
    mo.vstack(
        [
            mo.md("Rows with the context token they come from (shown here only)."),
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
            mo.md("As injected:"),
            mo.plain_text(fence(_j)),
        ]
    )
    return


@app.cell
def _(
    ITEMS,
    answer_ranks,
    band,
    intermediate_ranks,
    js,
    js_last,
    k,
    min_score,
    min_tf,
    results,
    summary,
):
    # Plain-text dump: an .ipynb export keeps printed text, but renders
    # tables as images.
    print("== intermediate: best rank @ token, per layer")
    for _row in intermediate_ranks:
        print(_row)
    print("\n== answer D: rank at the last position, per layer")
    for _row in answer_ranks:
        print(_row)
    print(
        f"\n== J: band={band.value} k={k.value} min_tf={min_tf.value} "
        f"min_score={min_score.value}"
    )
    for _row in summary:
        print(_row)
    for _item in ITEMS:
        _r = results[_item["slug"]]["readout"]
        print(f"\n-- {_item['slug']}")
        for _jrow in js[_item["slug"]]:
            print(
                f"{_r.context_tokens[_jrow.position]!r:>16}: {', '.join(_jrow.words)}"
            )
        print("   last position only:", [r.words for r in js_last[_item["slug"]]])
    return


if __name__ == "__main__":
    app.run()
