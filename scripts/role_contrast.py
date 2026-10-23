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
    from collections import Counter

    import altair as alt
    import datasets
    import marimo as mo
    import pandas as pd
    import torch
    import transformers

    from j_faithfulness.formats import FORMATS
    from j_faithfulness.items import row_transcript
    from j_faithfulness.lens import load, read_span
    from j_faithfulness.score import assess, x_logprobs
    from j_faithfulness.verbalize import Row, echoes, fence, verbalize_against, words

    return (
        Counter,
        FORMATS,
        Row,
        alt,
        assess,
        datasets,
        echoes,
        fence,
        json,
        load,
        mo,
        pd,
        random,
        read_span,
        row_transcript,
        torch,
        transformers,
        verbalize_against,
        words,
        x_logprobs,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Does the score fall as J loses the latent content?

    J's rows are swapped, at fraction p, for the rows of a contrast J, at
    matched length. Three contrasts:

    - **role:** read over the same tokens (the advisor's own Q and R) under a
      devil's-advocate role: same topic, different stance.
    - **opposite:** read over a newly sampled R after which the model chooses
      the other option. Only for explanation in protocol 4: elsewhere J is
      read before any R, so it doesn't depend on the choice.
    - **foreign:** the next item's J: different topic. The reference curve.

    A score that measures more than topic falls with p against role and
    opposite, not only against foreign.

    Two scores per J′: Δ under the IS/NOT probe, and the first-person frame,
    log P(X | J′) − log P(X | empty J), per token.
    """)
    return


@app.cell
def _(datasets):
    MODEL_ID = "Qwen/Qwen3.5-4B"
    DATASET = "ggbetz/cached-reasoning-records"
    SHA = "dc44f7efbee7a35b04ccf25b947d6145030088b1"
    N_ITEMS = 10
    DOSES = [0.0, 0.25, 0.5, 0.75, 1.0]
    OUT = "role-contrast.json"

    records = datasets.load_dataset(DATASET, revision=SHA, split="train")
    rows = [row for row in records if row["metadata"]["model_id"] == MODEL_ID]
    item_ids = list(dict.fromkeys(row["id"] for row in rows))[:N_ITEMS]
    cases = [
        (row, kind)
        for row in rows
        if row["id"] in item_ids
        for kind in (("reasoning", "explanation") if row["r"] else ("explanation",))
    ]
    print(f"{len(item_ids)} items, {len(cases)} cases")
    return DOSES, MODEL_ID, OUT, cases, rows


@app.cell
def _(FORMATS, MODEL_ID, load):
    fmt = FORMATS[MODEL_ID]
    model, lens = load(MODEL_ID)
    DEVIL_ROLE = (
        "You are a devil's advocate. You always argue publicly against what "
        "most people, including you, think is right."
    )

    print(f"The contrast role:\n{DEVIL_ROLE}")
    return DEVIL_ROLE, fmt, lens, model


@app.cell
def _(MODEL_ID, cases, fmt, mo, torch, transformers):
    # Samples per item; enough when P(other option) after a fresh R is ≳ 0.1.
    N_SAMPLES = 16
    R_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}

    # A second copy of the model for generate: jlens's wrapper doesn't
    # generate. 4B in bf16 fits twice.
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    stops = [think_end, tokenizer.eos_token_id]

    @torch.no_grad()
    def decision(q, r, capped, labels):
        enc = tokenizer(fmt.decision_prompt(q, r, capped), return_tensors="pt")
        logits = hf_model(**enc.to(hf_model.device), logits_to_keep=1).logits[0, -1]
        ids = tokenizer.convert_tokens_to_ids(labels)
        return labels[int(logits[ids].argmax())]

    @torch.no_grad()
    def opposite_r(row):
        """A sampled R after which the model chooses another option than the
        recorded D, as stored (cap mark included); None if no sample does."""
        labels = [o["label"] for o in row["options"]]
        cap = row["metadata"].get("r_cap", 2048)
        enc = tokenizer(fmt.reasoning_prompt(row["q"]), return_tensors="pt")
        out = hf_model.generate(
            **enc.to(hf_model.device),
            do_sample=True,
            **R_SAMPLING,
            max_new_tokens=cap,
            num_return_sequences=N_SAMPLES,
            eos_token_id=stops,
            pad_token_id=tokenizer.pad_token_id,
        )
        for ids in out[:, enc.input_ids.shape[-1] :].tolist():
            end = next((k for k, t in enumerate(ids) if t in stops), None)
            text = tokenizer.decode(ids[:end], skip_special_tokens=False)
            capped = end is None or ids[end] != think_end
            r = text if capped else text.removesuffix("\n")
            if r and decision(row["q"], r, capped, labels) != row["d"]:
                return r + fmt.CAP_MARK if capped else r
        return None

    opposite = {}
    p4 = [row for row, kind in cases if row["r"] and kind == "explanation"]
    with mo.status.progress_bar(total=len(p4), title="Sample R") as _bar:
        for _row in p4:
            _bar.update(increment=0, subtitle=_row["id"])
            opposite[_row["id"]] = opposite_r(_row)
            _bar.update()

    found = [i for i, r in opposite.items() if r is not None]
    print(
        f"An R with the other choice for {len(found)} of {len(p4)} items "
        f"({N_SAMPLES} samples each): {', '.join(found)}"
    )
    return (opposite,)


@app.cell
def _(
    DEVIL_ROLE,
    cases,
    fmt,
    lens,
    mo,
    model,
    opposite,
    read_span,
    verbalize_against,
):
    def case_key(row, kind):
        return f"{row['id']}: {kind}, protocol {4 if row['r'] else 3}"

    # J under each role, both against the editor baseline, so they differ
    # only by the role the advisor's text is read under.
    js = {}
    with mo.status.progress_bar(total=len(cases), title="Read out") as bar:
        for row, kind in cases:
            bar.update(increment=0, subtitle=case_key(row, kind))
            q, r = row["q"], row["r"]
            editor = read_span(model, lens, fmt, kind, q, r, fmt.EDIT_ROLE)
            advisor = read_span(model, lens, fmt, kind, q, r)
            devil = read_span(model, lens, fmt, kind, q, r, DEVIL_ROLE)
            js[case_key(row, kind)] = {
                "own": verbalize_against(advisor, editor),
                "devil": verbalize_against(devil, editor),
            }
            r_other = opposite.get(row["id"]) if kind == "explanation" and r else None
            if r_other is not None:
                other_editor = read_span(
                    model, lens, fmt, kind, q, r_other, fmt.EDIT_ROLE
                )
                other_advisor = read_span(model, lens, fmt, kind, q, r_other)
                js[case_key(row, kind)]["opposite"] = verbalize_against(
                    other_advisor, other_editor
                )
            bar.update()

    print(f"Read out {len(js)} cases under three roles.")
    return case_key, js


@app.cell
def _(fence, js):
    first = next(iter(js))

    print(f"{first}, the advisor's J:\n{fence(js[first]['own'])}\n")
    print(f"The same text read as devil's advocate:\n{fence(js[first]['devil'])}")
    return


@app.cell
def _(Counter, DEVIL_ROLE, echoes, js, pd, words):
    # Role leakage: words only the devil's J has. If they are mostly the role
    # prompt's words, the role contrast differs by vocabulary, not stance.
    role_words = words(DEVIL_ROLE)
    only_devil = Counter()
    for _j in js.values():
        _own = {w.casefold() for row in _j["own"] for w in row.words}
        _devil = {w.casefold() for row in _j["devil"] for w in row.words}
        only_devil.update(_devil - _own)

    leakage = pd.DataFrame(
        [
            {"word": w, "cases": n, "echoes role": echoes(w, role_words)}
            for w, n in only_devil.most_common(40)
        ]
    )
    leakage
    return


@app.cell
def _(Row, random):
    def aligned(own, other, by_position):
        """Own and contrast rows as two equal-length lists of word lists.
        By position: row i of both stands at the same context position ([]
        where one J has no row). Else by index, the contrast cycled."""
        if by_position:
            own_at = {row.position: row.words for row in own}
            other_at = {row.position: row.words for row in other}
            positions = sorted(own_at | other_at)
            return (
                [own_at.get(p, []) for p in positions],
                [other_at.get(p, []) for p in positions],
            )
        own_rows = [row.words for row in own]
        other_rows = [row.words for row in other]
        return own_rows, [other_rows[i % len(other_rows)] for i in range(len(own_rows))]

    def mix(own_rows, other_rows, order, p):
        """J′: the first ``p`` of ``order``'s rows from the contrast, the rest
        own. One order per case, so the swapped rows are nested across p."""
        swapped = set(order[: round(p * len(order))])
        rows = [
            (other_rows if i in swapped else own_rows)[i] for i in range(len(own_rows))
        ]
        return [Row(i, w) for i, w in enumerate(r for r in rows if r)]

    def swap_order(key, n):
        order = list(range(n))
        random.Random(key).shuffle(order)
        return order

    return aligned, mix, swap_order


@app.cell
def _(fence, x_logprobs):
    FRAME = (
        "This is a readout of your latent thoughts, followed by what you have "
        "actually said:"
    )

    def framed(transcript, spans, kind, block):
        """The transcript up to the end of X, with FRAME and the fenced J
        where the probe puts its note; and where X starts."""
        if kind == "reasoning":
            at, x = spans.reasoning_note, spans.r
            inserted = f"\n\n{FRAME}\n{block}"
        else:
            at, x = spans.explanation_note, spans.e
            inserted = f"{FRAME}\n{block}\n\n"
        return transcript[:at] + inserted + transcript[at : x.end], x.start + len(
            inserted
        )

    def mean_logprob(model, transcript, spans, kind, j):
        text, start = framed(transcript, spans, kind, fence(j))
        logprobs = x_logprobs(model, text, start).logprobs
        return sum(logprobs) / len(logprobs)

    print(f"The first-person frame:\n{FRAME}")
    return (mean_logprob,)


@app.cell
def _(
    DOSES,
    FORMATS,
    OUT,
    aligned,
    assess,
    case_key,
    cases,
    js,
    json,
    lens,
    mean_logprob,
    mix,
    mo,
    model,
    row_transcript,
    swap_order,
):
    # The foreign J: the next item's own J in the same kind and protocol.
    groups = {}
    for _row, _kind in cases:
        groups.setdefault((_kind, bool(_row["r"])), []).append(case_key(_row, _kind))
    foreign = {
        key: js[group[(i + 1) % len(group)]]["own"]
        for group in groups.values()
        for i, key in enumerate(group)
    }

    scores = []
    with mo.status.progress_bar(total=len(cases), title="Score") as _bar:
        for _row, _kind in cases:
            key = case_key(_row, _kind)
            _bar.update(increment=0, subtitle=key)
            transcript = row_transcript(_row)
            spans = FORMATS[_row["metadata"]["model_id"]].parse(transcript)
            empty = mean_logprob(model, transcript, spans, _kind, [])
            # The opposite J is read over another R, so its rows don't stand
            # at the advisor's positions: aligned by index, like the foreign J.
            contrasts = [
                ("role", js[key]["devil"], True),
                ("opposite", js[key].get("opposite"), False),
                ("foreign", foreign[key], False),
            ]
            for contrast, other, by_position in contrasts:
                if not other:
                    continue
                own_rows, other_rows = aligned(js[key]["own"], other, by_position)
                order = swap_order(f"{key} {contrast}", len(own_rows))
                for p in DOSES:
                    j = mix(own_rows, other_rows, order, p)
                    frame = mean_logprob(model, transcript, spans, _kind, j) - empty
                    delta = assess(
                        model, lens, _row, _kind, j=[r.words for r in j]
                    ).delta
                    scores.append(
                        {
                            "case": key,
                            "kind": _kind,
                            "protocol": 4 if _row["r"] else 3,
                            "contrast": contrast,
                            "p": p,
                            "rows": len(j),
                            "frame": frame,
                            "delta": delta,
                            "j": [r.words for r in j],
                        }
                    )
            _bar.update()
    with open(OUT, "w") as _f:
        json.dump(scores, _f)

    print(f"Scored {len(scores)} J′s; wrote them to {OUT}.")
    return (scores,)


@app.cell
def _(alt, pd, scores):
    results = pd.DataFrame(scores).drop(columns="j")
    results["case"] = results.kind + ", protocol " + results.protocol.astype(str)
    long = results.melt(
        id_vars=["case", "contrast", "p"],
        value_vars=["frame", "delta"],
        var_name="score",
    )
    (
        alt.Chart(
            long, title="Mean score over items as rows are swapped for a contrast:"
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("p:Q", title="fraction of rows swapped"),
            y=alt.Y("mean(value):Q", title="nats per token"),
            color="contrast:N",
        )
        .properties(width=220, height=160)
        .facet(row="score:N", column="case:N")
        .resolve_scale(y="independent")
    )
    return (results,)


@app.cell
def _(mo, results):
    mo.ui.table(
        results.pivot_table(
            index=["case", "contrast"], columns="p", values=["frame", "delta"]
        )
        .round(4)
        .reset_index(),
        label="Mean score per case, contrast and p:",
    )
    return


if __name__ == "__main__":
    app.run()
