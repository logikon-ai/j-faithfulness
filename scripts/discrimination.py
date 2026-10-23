# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "altair",
#     "datasets",
#     "transformers>=5.5",
#     "flash-linear-attention",
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
    import transformers

    from j_faithfulness.formats import FORMATS

    return FORMATS, alt, datasets, json, mo, pd, random, torch, transformers


@app.cell
def _(mo):
    mo.md(r"""
    # Can the model tell which readout belongs to a rationale?

    The model sees X (R or E) and two readouts, and answers A or B: one
    forward pass per question. The right candidate is X's own J, the wrong
    one another item's J. Both are diluted the same way: a fraction p of
    their rows, at the same places, is replaced by the rows of a third
    item's J. The signal fraction 1 − p sets the difficulty; at p = 1 both
    candidates are the same and accuracy is at chance.

    Each question is asked in both orders, and the log-odds of the right
    answer are averaged over them, which cancels a bias towards A or B.
    """)
    return


@app.cell
def _(datasets, json):
    MODEL_ID = "Qwen/Qwen3.5-4B"
    DATASET = "ggbetz/cached-reasoning-records"
    SHA = "dc44f7efbee7a35b04ccf25b947d6145030088b1"
    # J's from a run of demo.py (cells 17–24): no read-outs here.
    J_FILE = "assessments.json"
    DOSES = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    # Distractor and filler pairs per case: 10 items give little data, and
    # different pairs average out how unlike two particular items are.
    DRAWS = 4
    OUT = "discrimination.json"

    with open(J_FILE) as f:
        assessments = json.load(f)
    records = datasets.load_dataset(DATASET, revision=SHA, split="train")
    xs = {
        (row["id"], 4 if row["r"] else 3, kind): row["r"]
        if kind == "reasoning"
        else row["e"]
        for row in records
        if row["metadata"]["model_id"] == MODEL_ID
        for kind in ("reasoning", "explanation")
    }
    cases = [
        {
            "id": a["id"],
            "protocol": a["protocol"],
            "kind": a["kind"],
            "x": xs[a["id"], a["protocol"], a["kind"]],
            "j": a["j"],
        }
        for a in assessments
    ]

    print(f"{len(cases)} cases from {J_FILE}, {len({c['id'] for c in cases})} items")
    return DOSES, DRAWS, MODEL_ID, OUT, cases


@app.cell
def _(MODEL_ID, torch, transformers):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Load {MODEL_ID}.")
    return hf_model, tokenizer


@app.cell
def _(FORMATS, MODEL_ID):
    fmt = FORMATS[MODEL_ID]

    def block(rows):
        return "\n".join(", ".join(words) + "." for words in rows)

    def question(kind, x, rows_a, rows_b):
        """The prompt, raw ChatML in the transcripts' format, ending where
        the answer letter goes. Thinking off: one forward pass."""
        user = (
            f"Below is an assistant's {kind}, and two readouts of an "
            "assistant's latent thoughts, one line of key words per word or "
            "sentence read. One readout was taken in the conversation where "
            f"the assistant wrote this {kind}, the other in another "
            "conversation. Which readout belongs to the "
            f"{kind}? Answer with A or B.\n\n"
            f"{kind.capitalize()}:\n{x}\n\n"
            f"Readout A:\n{block(rows_a)}\n\n"
            f"Readout B:\n{block(rows_b)}"
        )
        return f"{fmt.USER_OPEN}{user}{fmt.TURN_END}\n{fmt.EMPTY_THINK}Answer:"

    print(
        question("explanation", "<E>", [["w1", "w2"], ["w3"]], [["v1"], ["v2", "v3"]])
    )
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

    def diluted(own, foreign, filler, p, seed):
        """The right and the wrong candidate at dose ``p``: all three J's cut
        to one length, then the same ``p`` of rows in both replaced by the
        filler's. One order per seed, so the replaced rows nest across p."""
        n = min(len(own), len(foreign), len(filler))
        own, foreign, filler = (spread_rows(rows, n) for rows in (own, foreign, filler))
        order = list(range(n))
        random.Random(seed).shuffle(order)
        replaced = set(order[: round(p * n)])
        right = [filler[i] if i in replaced else own[i] for i in range(n)]
        wrong = [filler[i] if i in replaced else foreign[i] for i in range(n)]
        return right, wrong

    return (diluted,)


@app.cell
def _(hf_model, tokenizer, torch):
    letters = {}
    for _letter in "AB":
        _ids = tokenizer.encode(f" {_letter}", add_special_tokens=False)
        assert len(_ids) == 1, f"' {_letter}' is not one token"
        letters[_letter] = _ids[0]

    @torch.no_grad()
    def log_odds_a(prompt):
        """log P(A) − log P(B) for the next token."""
        enc = tokenizer(prompt, return_tensors="pt").to(hf_model.device)
        logits = hf_model(**enc, logits_to_keep=1).logits[0, -1].float()
        return float(logits[letters["A"]] - logits[letters["B"]])

    return (log_odds_a,)


@app.cell
def _(DOSES, DRAWS, OUT, cases, diluted, json, log_odds_a, mo, question, random):
    # Distractor m and filler k from the same kind and protocol, so all
    # candidates were read the same way.
    groups = {}
    for _i, _case in enumerate(cases):
        groups.setdefault((_case["kind"], _case["protocol"]), []).append(_i)

    trials = []
    with mo.status.progress_bar(
        total=len(cases) * DRAWS * len(DOSES), title="Ask"
    ) as _bar:
        for _i, _case in enumerate(cases):
            others = [i for i in groups[_case["kind"], _case["protocol"]] if i != _i]
            pairs = random.Random(_i).sample(
                [(m, k) for m in others for k in others if m != k], DRAWS
            )
            for _draw, (m, k) in enumerate(pairs):
                for p in DOSES:
                    _bar.update(
                        increment=0, subtitle=f"{_case['id']} {_case['kind']}, p = {p}"
                    )
                    right, wrong = diluted(
                        _case["j"], cases[m]["j"], cases[k]["j"], p, f"{_i} {_draw}"
                    )
                    right_first = log_odds_a(
                        question(_case["kind"], _case["x"], right, wrong)
                    )
                    right_second = -log_odds_a(
                        question(_case["kind"], _case["x"], wrong, right)
                    )
                    trials.append(
                        {
                            "id": _case["id"],
                            "case": f"{_case['kind']}, protocol {_case['protocol']}",
                            "foreign": cases[m]["id"],
                            "filler": cases[k]["id"],
                            "p": p,
                            "rows": len(right),
                            "right_first": right_first,
                            "right_second": right_second,
                        }
                    )
                    _bar.update()
    with open(OUT, "w") as _f:
        json.dump(trials, _f)

    print(f"Asked {2 * len(trials)} questions; wrote them to {OUT}.")
    return (trials,)


@app.cell
def _(pd, trials):
    results = pd.DataFrame(trials)
    # Averaged over both orders, the log-odds of the right answer; > 0 counts
    # as correct. Per order, the raw accuracy shows the position bias.
    results["log_odds"] = (results.right_first + results.right_second) / 2
    results["correct"] = results.log_odds > 0
    results["correct_first"] = results.right_first > 0
    results["correct_second"] = results.right_second > 0
    results["signal"] = 1 - results.p

    results.groupby(["case", "signal"])[
        ["correct", "correct_first", "correct_second", "log_odds"]
    ].mean().round(3)
    return (results,)


@app.cell
def _(alt, results):
    _base = alt.Chart(results).encode(
        x=alt.X("signal:Q", title="signal fraction (1 − p)"),
        color=alt.Color("case:N", title=None),
    )
    (
        _base.mark_line(point=True)
        .encode(
            y=alt.Y("mean(correct):Q", title="accuracy", scale=alt.Scale(domain=[0, 1]))
        )
        .properties(title="Accuracy as the candidates are diluted:", width=320)
        | _base.mark_line(point=True)
        .encode(y=alt.Y("mean(log_odds):Q", title="log-odds of the right answer"))
        .properties(title="Mean log-odds, averaged over both orders:", width=320)
    )
    return


@app.cell
def _(pd, results):
    def threshold(group, level=0.75):
        """The lowest signal fraction at which accuracy reaches ``level``,
        linearly interpolated; None if it never does."""
        acc = group.groupby("signal").correct.mean().sort_index()
        above = acc[acc >= level]
        if above.empty:
            return None
        s1 = above.index[0]
        below = acc[acc.index < s1]
        if below.empty:
            return s1
        s0, a0, a1 = below.index[-1], below.iloc[-1], acc[s1]
        return s0 + (level - a0) * (s1 - s0) / (a1 - a0)

    pd.DataFrame(
        {
            "accuracy at full signal": results[results.signal == 1]
            .groupby("case")
            .correct.mean(),
            "signal for 75%": results.groupby("case").apply(threshold),
        }
    ).round(3)
    return


if __name__ == "__main__":
    app.run()
