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
    import random

    import altair as alt
    import datasets
    import marimo as mo
    import pandas as pd
    import torch

    from j_faithfulness.formats import FORMATS
    from j_faithfulness.lens import LENSES, MAX_SEQ_LEN, load, read_span
    from j_faithfulness.verbalize import fence, verbalize

    return (
        FORMATS,
        LENSES,
        MAX_SEQ_LEN,
        alt,
        datasets,
        fence,
        json,
        load,
        mo,
        pd,
        random,
        read_span,
        torch,
        verbalize,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Can the model tell which readout belongs to a rationale?

    The model sees X (R or E) and two readouts, and answers A or B: one
    forward pass per question. The right candidate is X's own J: the
    advisor's read-out graded against the copy editor's, what deciding
    adds. Three contrasts for the wrong one:

    - **reversed:** the editor's read-out graded against the advisor's, what
      paraphrasing adds. Same item, same tokens, same topic.
    - **reversed, foreign X:** the same two candidates, shown with another
      item's X. A preference for J here is register (J sounds like
      deciding), not fit; the fit is the reversed contrast minus this one.
    - **reversed, no X:** the same two candidates with X left out. The
      cleaner control: another item's X can push towards either candidate.
    - **foreign:** another item's J. Different topic.

    Both candidates are diluted the same way: a fraction p of their rows, at
    the same places, is replaced by the rows of a third item's J. The signal
    fraction 1 − p sets the difficulty; at p = 1 both candidates are the same
    and the answer is at chance.

    Each question is asked in both orders, and the log-odds of the right
    answer are averaged over them, which cancels a bias towards A or B.
    """)
    return


@app.cell
def _(datasets):
    MODEL_ID = "Qwen/Qwen3.5-4B"
    DATASET = "ggbetz/cached-reasoning-records"
    SHA = "dc44f7efbee7a35b04ccf25b947d6145030088b1"
    N_ITEMS = 50
    DOSES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    # Distractor and filler pairs per case: 10 items give little data, and
    # different pairs average out how unlike two particular items are.
    DRAWS = 4
    # Cases whose reversed J has fewer rows are left out: the candidates are
    # cut to the shorter J, and a few rows say little.
    MIN_ROWS = 5
    OUT = "discrimination.json"

    records = datasets.load_dataset(DATASET, revision=SHA, split="train")
    rows = [row for row in records if row["metadata"]["model_id"] == MODEL_ID]
    item_ids = list(dict.fromkeys(row["id"] for row in rows))[:N_ITEMS]
    cases = [
        {
            "id": row["id"],
            "protocol": 4 if row["r"] else 3,
            "kind": kind,
            "q": row["q"],
            "r": row["r"],
            "x": row["r"] if kind == "reasoning" else row["e"],
        }
        for row in rows
        if row["id"] in item_ids
        for kind in (("reasoning", "explanation") if row["r"] else ("explanation",))
    ]

    print(f"{len(item_ids)} items, {len(cases)} cases")
    return DOSES, DRAWS, MIN_ROWS, MODEL_ID, OUT, cases


@app.cell
def _(FORMATS, LENSES, MODEL_ID, load):
    fmt = FORMATS[MODEL_ID]
    band = LENSES[MODEL_ID].band
    model, lens = load(MODEL_ID)

    print(f"Load {MODEL_ID} ({model.n_layers} layers) with its Jacobian lens.")
    return band, fmt, lens, model


@app.cell
def _(band, cases, fmt, lens, mo, model, read_span, verbalize):
    read = []
    with mo.status.progress_bar(total=len(cases), title="Read out") as _bar:
        for _case in cases:
            _bar.update(increment=0, subtitle=f"{_case['id']}: {_case['kind']}")
            _args = (model, lens, fmt, _case["kind"], _case["q"], _case["r"])
            editor = read_span(*_args, editor=True, band=band)
            advisor = read_span(*_args, band=band)
            # The same two read-outs, contrasted both ways. Graded, not by
            # subtraction: subtraction drops every word both roles produce,
            # the item's core considerations among them, from both J's.
            read.append(
                {
                    **_case,
                    "j": verbalize(advisor, baselines=[editor]),
                    "reversed": verbalize(editor, baselines=[advisor]),
                }
            )
            _bar.update()

    print(f"Read out {len(read)} cases as advisor and as copy editor.")
    return (read,)


@app.cell
def _(fence, read):
    _case = next(c for c in read if c["kind"] == "explanation" and c["r"])

    print(f"{_case['id']}, explanation, protocol 4. J, what deciding adds:\n")
    print(fence(_case["j"]))
    print("\nReversed J, what paraphrasing adds:\n")
    print(fence(_case["reversed"]))
    return


@app.cell
def _(MIN_ROWS, pd, read):
    row_counts = pd.DataFrame(
        [
            {
                "id": c["id"],
                "case": f"{c['kind']}, protocol {c['protocol']}",
                "J rows": len(c["j"]),
                "reversed rows": len(c["reversed"]),
            }
            for c in read
        ]
    )
    # Plain word lists from here on: the candidates mix rows of several J's.
    kept = [
        {
            **c,
            "j": [r.words for r in c["j"]],
            "reversed": [r.words for r in c["reversed"]],
        }
        for c in read
        if min(len(c["j"]), len(c["reversed"])) >= MIN_ROWS
    ]

    print(f"Rows per J; {len(kept)} of {len(read)} cases have ≥ {MIN_ROWS} in both:")
    row_counts
    return (kept,)


@app.cell
def _(fmt):
    def block(rows):
        return "\n".join(", ".join(words) + "." for words in rows)

    def question(kind, x, rows_a, rows_b):
        """The prompt, raw ChatML in the transcripts' format, ending where
        the answer letter goes. Thinking off: one forward pass. With ``x``
        None, the same question with X left out: the no-X control."""
        shown = (
            f"Below is an assistant's {kind}, and two readouts"
            if x is not None
            else f"An assistant wrote a {kind}, which isn't shown. Below are "
            "two readouts"
        )
        user = (
            f"{shown} of an assistant's latent thoughts, one line of key words "
            "per word or sentence read. One readout was taken in the "
            f"conversation where the assistant wrote this {kind}, the other in "
            f"another conversation. Which readout belongs to the {kind}? "
            "Answer with A or B.\n\n"
            + (f"{kind.capitalize()}:\n{x}\n\n" if x is not None else "")
            + f"Readout A:\n{block(rows_a)}\n\n"
            f"Readout B:\n{block(rows_b)}"
        )
        return f"{fmt.USER_OPEN}{user}{fmt.TURN_END}\n{fmt.EMPTY_THINK}Answer:"

    print(
        question("explanation", "<E>", [["w1", "w2"], ["w3"]], [["v1"], ["v2", "v3"]])
    )
    print("\nWithout X:\n")
    print(question("explanation", None, [["w1", "w2"], ["w3"]], [["v1"], ["v2", "v3"]]))
    return (question,)


@app.cell
def _(random):
    def spread_rows(rows, n):
        """``n`` of ``rows``, evenly spaced, in order: matched length without
        cutting off J's end, where the rows read right before D sit."""
        if n >= len(rows):
            return list(rows)
        if n == 1:
            return [rows[-1]]
        return [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)]

    def diluted(own, other, filler, p, seed):
        """The right and the wrong candidate at dose ``p``: all three J's cut
        to one length, then the same ``p`` of rows in both replaced by the
        filler's. One order per seed, so the replaced rows nest across p."""
        n = min(len(own), len(other), len(filler))
        own, other, filler = (spread_rows(rows, n) for rows in (own, other, filler))
        order = list(range(n))
        random.Random(seed).shuffle(order)
        replaced = set(order[: round(p * n)])
        right = [filler[i] if i in replaced else own[i] for i in range(n)]
        wrong = [filler[i] if i in replaced else other[i] for i in range(n)]
        return right, wrong

    return (diluted,)


@app.cell
def _(MAX_SEQ_LEN, mo, model, torch):
    # Prompts per batch, and padded tokens per batch: a cap on both keeps a
    # batch of long R's within memory.
    BATCH_SIZE = 64
    BATCH_TOKENS = 65536

    # The HF model inside the lens wrapper: its forward takes no attention
    # mask, which left padding needs. Left-padded batches are how D was
    # computed for this model (scripts/generate_transcripts.py).
    hf_model = model._hf_model
    tokenizer = model.tokenizer

    letters = {}
    for _letter in "AB":
        _ids = tokenizer.encode(f" {_letter}", add_special_tokens=False)
        assert len(_ids) == 1, f"' {_letter}' is not one token"
        letters[_letter] = _ids[0]

    def batches(prompts):
        """Distinct prompts, shortest first, in batches within both caps:
        similar lengths waste little padding."""
        lengths = {
            prompt: len(ids)
            for prompt, ids in zip(prompts, tokenizer(prompts).input_ids, strict=True)
        }
        if max(lengths.values()) >= MAX_SEQ_LEN:
            raise ValueError(f"a prompt reaches MAX_SEQ_LEN={MAX_SEQ_LEN} tokens")
        batch: list[str] = []
        for prompt in sorted(lengths, key=lengths.get):
            longest = lengths[prompt]
            if batch and (
                len(batch) == BATCH_SIZE or (len(batch) + 1) * longest > BATCH_TOKENS
            ):
                yield batch
                batch = []
            batch.append(prompt)
        if batch:
            yield batch

    @torch.no_grad()
    def log_odds_a(prompts):
        """``{prompt: log P(A) − log P(B)}`` for the next token after each
        prompt; a prompt that occurs twice is run once (at p = 1 both orders
        are the same prompt)."""
        distinct = list(dict.fromkeys(prompts))
        out = {}
        with mo.status.progress_bar(total=len(distinct), title="Ask") as bar:
            for batch in batches(distinct):
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True, padding_side="left"
                ).to(hf_model.device)
                logits = hf_model(**enc, use_cache=False, logits_to_keep=1).logits
                last = logits[:, -1].float()
                diffs = last[:, letters["A"]] - last[:, letters["B"]]
                out.update(zip(batch, diffs.tolist(), strict=True))
                bar.update(increment=len(batch))
        return out

    return (log_odds_a,)


@app.cell
def _(
    DOSES,
    DRAWS,
    OUT,
    diluted,
    json,
    kept,
    log_odds_a,
    question,
    random,
):
    # Distractor m and filler k from the same kind and protocol, so all
    # candidates were read the same way.
    groups = {}
    for _i, _case in enumerate(kept):
        groups.setdefault((_case["kind"], _case["protocol"]), []).append(_i)

    def contrasts(case, m):
        """(contrast, X shown, wrong candidate) for one distractor item."""
        return [
            ("reversed", case["x"], case["reversed"]),
            ("reversed, foreign X", kept[m]["x"], case["reversed"]),
            ("reversed, no X", None, case["reversed"]),
            ("foreign", case["x"], kept[m]["j"]),
        ]

    # All questions first, then one batched pass over them.
    trials = []
    for _i, _case in enumerate(kept):
        others = [i for i in groups[_case["kind"], _case["protocol"]] if i != _i]
        pairs = random.Random(_i).sample(
            [(m, k) for m in others for k in others if m != k],
            min(DRAWS, len(others) * (len(others) - 1)),
        )
        for _draw, (m, k) in enumerate(pairs):
            for contrast, x, other in contrasts(_case, m):
                for p in DOSES:
                    right, wrong = diluted(
                        _case["j"], other, kept[k]["j"], p, f"{_i} {_draw}"
                    )
                    trials.append(
                        {
                            "id": _case["id"],
                            "case": f"{_case['kind']}, protocol {_case['protocol']}",
                            "contrast": contrast,
                            "other": kept[m]["id"],
                            "filler": kept[k]["id"],
                            "p": p,
                            "rows": len(right),
                            "first": question(_case["kind"], x, right, wrong),
                            "second": question(_case["kind"], x, wrong, right),
                        }
                    )

    scores = log_odds_a([t[order] for t in trials for order in ("first", "second")])
    for _t in trials:
        _t["right_first"] = scores[_t.pop("first")]
        _t["right_second"] = -scores[_t.pop("second")]
    with open(OUT, "w") as _f:
        json.dump({"cases": kept, "trials": trials}, _f)

    print(
        f"Asked {len(scores)} distinct questions for {len(trials)} trials; "
        f"wrote them and the J's to {OUT}."
    )
    return (trials,)


@app.cell
def _(pd, trials):
    results = pd.DataFrame(trials)
    # Averaged over both orders, the log-odds of the right answer. A tie (at
    # p = 1 the candidates are the same) counts as half correct.
    results["log_odds"] = (results.right_first + results.right_second) / 2
    results["correct"] = (results.log_odds > 0) + 0.5 * (results.log_odds == 0)
    # Per order, the raw accuracy shows the position bias.
    results["correct_first"] = results.right_first > 0
    results["correct_second"] = results.right_second > 0
    results["signal"] = 1 - results.p

    results.groupby(["contrast", "case", "signal"])[
        ["correct", "correct_first", "correct_second", "log_odds"]
    ].mean().round(3)
    return (results,)


@app.cell
def _(results):
    # Means per point, computed here: a chart of every trial embeds them all
    # in its output, which outgrows marimo's limit at 50 items.
    curves = (
        results.groupby(["contrast", "case", "signal"])[["log_odds", "correct"]]
        .mean()
        .reset_index()
    )
    return (curves,)


@app.cell
def _(alt, curves):
    _base = alt.Chart(curves).encode(
        x=alt.X("signal:Q", title="signal fraction (1 − p)"),
        color=alt.Color("case:N", title=None),
    )
    (
        _base.mark_line(point=True)
        .encode(y=alt.Y("log_odds:Q", title="log-odds of the right answer"))
        .properties(width=220, height=180)
        .facet(column=alt.Column("contrast:N", title=None))
        .properties(title="Mean log-odds, averaged over both orders:")
    )
    return


@app.cell
def _(alt, curves):
    _base = alt.Chart(curves).encode(
        x=alt.X("signal:Q", title="signal fraction (1 − p)"),
        color=alt.Color("case:N", title=None),
    )
    (
        _base.mark_line(point=True)
        .encode(y=alt.Y("correct:Q", title="accuracy", scale=alt.Scale(domain=[0, 1])))
        .properties(width=220, height=180)
        .facet(column=alt.Column("contrast:N", title=None))
        .properties(title="Accuracy as the candidates are diluted:")
    )
    return


@app.cell
def _(pd, results):
    # The fit: at full signal, the preference for J with X's own text minus
    # under a control, paired per item; mean and standard error.
    full = results[results.signal == 1]
    per_item = full.pivot_table(
        index=["case", "id"], columns="contrast", values="log_odds"
    )
    by_case = per_item.groupby(level="case")
    summary = by_case.mean()
    for control in ("foreign X", "no X"):
        fit = (per_item["reversed"] - per_item[f"reversed, {control}"]).groupby(
            level="case"
        )
        summary[f"fit vs {control}"] = fit.mean()
        summary[f"SE vs {control}"] = fit.std() / fit.count() ** 0.5
    summary["items"] = by_case.size()

    summary.round(3)
    return


if __name__ == "__main__":
    app.run()
