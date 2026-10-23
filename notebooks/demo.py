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

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import dataclasses
    import json

    import altair as alt
    import datasets
    import marimo as mo
    import pandas as pd
    import transformers

    from j_faithfulness.formats import FORMATS
    from j_faithfulness.items import row_transcript
    from j_faithfulness.lens import BAND, LENS_FILE, load, read_span
    from j_faithfulness.probe import probe
    from j_faithfulness.score import (
        against_control,
        assess,
        case,
        deltas,
        j_word_gaps,
        mismatched,
        per_sentence,
        spread,
        x_logprobs,
    )
    from j_faithfulness.verbalize import document, fence, verbalize_against

    return (
        BAND,
        FORMATS,
        LENS_FILE,
        against_control,
        alt,
        assess,
        case,
        dataclasses,
        datasets,
        deltas,
        document,
        fence,
        j_word_gaps,
        json,
        load,
        mismatched,
        mo,
        pd,
        per_sentence,
        probe,
        read_span,
        row_transcript,
        spread,
        transformers,
        verbalize_against,
        x_logprobs,
    )


@app.cell
def _(datasets):
    MODEL_ID = "Qwen/Qwen3.5-4B"
    DATASET = "ggbetz/cached-reasoning-records"
    SHA = "dc44f7efbee7a35b04ccf25b947d6145030088b1"
    ITEM_IDX = 17

    records = datasets.load_dataset(DATASET, revision=SHA, split="train")
    # Protocol 4: the rows with an R.
    record = [
        row
        for row in records
        if row["r"] is not None and row["metadata"]["model_id"] == MODEL_ID
    ][ITEM_IDX]
    d_probs = ", ".join(
        f"P({option['label']}) = {p:.2f}"
        for option, p in zip(record["options"], record["d_probs"], strict=True)
    )

    print(
        f"Consider a reasoning trace generated with {MODEL_ID} "
        f"(cached at {DATASET}@{SHA[:7]}):\n"
    )
    print(f"[[Query (Q):]]\n{record['q']}\n")
    print(f"[[Reasoning (R):]]\n{record['r']}\n")
    print(f"[[Decision (D):]]\n{record['d']}  ({d_probs})\n")
    print(f"[[Explanation (E):]]\n{record['e']}")
    return MODEL_ID, record, records


@app.cell
def _(alt):
    def strip(tokens, color, title, points=None):
        """The whole transcript as one row of tokens: those with a ``value``
        colored by ``color``, the rest light grey; a labelled rule at each of
        ``points``."""
        base = (
            alt.Chart(tokens, title=title)
            .encode(x=alt.X("i:O", axis=None), tooltip=list(tokens.columns))
            .properties(width=800, height=40)
        )
        chart = base.mark_rect(color="lightgrey") + base.transform_filter(
            "isValid(datum.value)"
        ).mark_rect().encode(color=color)
        if points is not None:
            at_point = alt.Chart(points).encode(x="i:O")
            chart += at_point.mark_rule(size=2) + at_point.mark_text(
                align="left", dx=3, dy=-26
            ).encode(text="label")
        return chart.configure_legend(orient="bottom")

    return (strip,)


@app.cell
def _(FORMATS, MODEL_ID, alt, pd, record, row_transcript, strip, transformers):
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    transcript = row_transcript(record)
    spans = FORMATS[MODEL_ID].parse(transcript)
    encoding = tokenizer(transcript, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    parts = {
        "q": "Query (Q)",
        "r": "Reasoning (R)",
        "d": "Decision (D)",
        "e": "Explanation (E)",
    }

    def token_at(char):
        return next(i for i, (_, end) in enumerate(offsets) if end > char)

    tokens = pd.DataFrame(
        {
            "i": range(len(offsets)),
            "token": [tokenizer.decode([t]) for t in encoding["input_ids"]],
            "value": [parts.get(spans.part(start)) for start, _ in offsets],
        }
    )
    points = pd.DataFrame(
        {
            # J's span ends with the token before the point.
            "i": [
                token_at(spans.reasoning_point) - 1,
                token_at(spans.explanation_point) - 1,
            ],
            "label": ["J before R", "J before D"],
        }
    )

    strip(
        tokens,
        alt.Color("value:N", sort=list(parts.values()), title=None),
        "Where faithfulness is tested: R against J before R, E against J before D.",
        points,
    )
    return offsets, spans, tokenizer, tokens, transcript


@app.cell
def _(LENS_FILE, MODEL_ID, load):
    model, lens = load(MODEL_ID)

    print(
        f"Load {MODEL_ID} ({model.n_layers} layers) with the Jacobian lens {LENS_FILE}."
    )
    return lens, model


@app.cell
def _(FORMATS, MODEL_ID, record):
    fmt = FORMATS[MODEL_ID]
    editor_context = fmt.reasoning_prompt(record["q"], role=fmt.EDIT_ROLE)

    print("The transcript as the model reads it when acting as copy editor:\n")
    print(editor_context)
    return (fmt,)


@app.cell
def _(alt, document, pd):
    def top_words(readout):
        """The lens's best word-like token at each word and layer."""
        return pd.DataFrame(
            [
                {
                    "word": f"{n:>3} {readout.context_tokens[position]!r}",
                    "layer": layer,
                    "top": next(iter(document(ranked).values()), (0, ""))[1],
                }
                for n, (position, layers) in enumerate(
                    zip(readout.positions, readout.top, strict=True)
                )
                for layer, ranked in zip(readout.layers, layers, strict=True)
            ]
        )

    def grid(readout, title, baseline=None):
        """Words (rows) × layers (columns) of a read-out; with a baseline,
        the top words it lacks stand out in color."""
        cells = top_words(readout)
        if baseline is None:
            cells["new"] = False
        else:
            seen = set(top_words(baseline).top.str.casefold())
            cells["new"] = ~cells.top.str.casefold().isin(seen)
        cells["top"] = cells.top.map(lambda w: w if len(w) <= 10 else w[:9] + "…")
        return (
            alt.Chart(cells, title=title)
            .mark_text()
            .encode(
                x="layer:O",
                y=alt.Y("word:N", sort=None, title=None),
                text="top",
                color=alt.condition(
                    "datum.new", alt.value("crimson"), alt.value("black")
                ),
            )
            .properties(width=alt.Step(60))
        )

    return (grid,)


@app.cell
def _(BAND, fmt, grid, lens, model, read_span, record):
    editor = read_span(model, lens, fmt, "reasoning", record["q"], role=fmt.EDIT_ROLE)

    grid(editor, f"The lens's top words at each word of Q, layers {BAND[0]}–{BAND[1]}:")
    return (editor,)


@app.cell
def _(fmt, record):
    advisor_context = fmt.reasoning_prompt(record["q"])

    print("The transcript as the model reads it when acting as advisor:\n")
    print(advisor_context)
    return


@app.cell
def _(fmt, grid, lens, model, read_span, record, editor):
    advisor = read_span(model, lens, fmt, "reasoning", record["q"])

    grid(
        advisor,
        "The same text, read to decide it; in red, words the editor's read-out lacks:",
        editor,
    )
    return (advisor,)


@app.cell
def _(advisor, fence, editor, verbalize_against):
    j_reasoning = verbalize_against(advisor, editor)

    print("What deciding adds over paraphrasing, as J:\n")
    print(fence(j_reasoning))
    return (j_reasoning,)


@app.cell
def _(fence, j_reasoning, probe, spans, transcript):
    probed = {
        based: probe(transcript, spans, "reasoning", fence(j_reasoning), based)
        for based in (True, False)
    }

    is_text, is_x = probed[True]
    not_text, _ = probed[False]
    # Offsets into the probed texts, which start with the archive header.
    is_note = is_text.index("[Re-analysis")
    not_note = not_text.index("[Re-analysis")

    print("The two contexts R is scored in; they differ only in NOT:\n")
    print(is_text[: is_text.index("<|im_start|>")].rstrip() + "\n")
    print(f"...{is_text[is_note - 40 : is_x.start]}...\n")
    print("and the same with the note:\n")
    print(not_text[not_note : not_text.index("<latentthought>")].strip())
    return (probed,)


@app.cell
def _(deltas, model, probed, x_logprobs):
    r_based, r_not_based = (
        x_logprobs(model, text, x.start) for text, x in probed.values()
    )
    delta_r = deltas(r_based, r_not_based)

    print(
        f"R is {sum(delta_r) / len(delta_r):+.3f} nats per token more likely "
        "when it is said to be based on J."
    )
    return delta_r, r_based


@app.cell
def _(alt, offsets, per_sentence, strip, tokenizer, tokens):
    def profile_strip(probed_arm, x, logprobs, values, title):
        """Δ per sentence of X on the whole transcript, X at span ``x``: red
        where X is less likely when said to be based on J, blue where more."""
        # The scored tokens are the last of the probed text; shift their
        # offsets back into the transcript to find X's tokens there.
        probed_text, probed_x = probed_arm
        scored = tokenizer(probed_text, return_offsets_mapping=True)["offset_mapping"]
        shift = probed_x.start - x.start
        delta_at = {
            start - shift: delta
            for (start, _), delta in zip(
                scored[-len(values) :],
                per_sentence(logprobs.tokens, values),
                strict=True,
            )
        }
        profile = tokens.assign(value=[delta_at.get(start) for start, _ in offsets])
        # One scale for every strip, symmetric around 0: the sign shows as
        # red or blue, and R and E compare by eye.
        return strip(
            profile,
            alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="redblue", domain=[-0.15, 0.15], clamp=True),
                title="Δ",
            ),
            title,
        )

    return (profile_strip,)


@app.cell
def _(delta_r, probed, profile_strip, r_based, spans):
    profile_strip(
        probed[True],
        spans.r,
        r_based,
        delta_r,
        "Where in R the fit sits, sentence by sentence:",
    )
    return


@app.cell
def _(fence, fmt, lens, model, read_span, record, verbalize_against):
    e_editor = read_span(
        model, lens, fmt, "explanation", record["q"], record["r"], fmt.EDIT_ROLE
    )
    e_advisor = read_span(model, lens, fmt, "explanation", record["q"], record["r"])
    j_explanation = verbalize_against(e_advisor, e_editor)

    print("J right before the decision, read over Q and R:\n")
    print(fence(j_explanation))
    return (j_explanation,)


@app.cell
def _(
    deltas,
    fence,
    j_explanation,
    model,
    probe,
    profile_strip,
    spans,
    transcript,
    x_logprobs,
):
    e_probed = [
        probe(transcript, spans, "explanation", fence(j_explanation), based)
        for based in (True, False)
    ]
    e_based, e_not_based = (x_logprobs(model, text, x.start) for text, x in e_probed)
    delta_e = deltas(e_based, e_not_based)

    profile_strip(
        e_probed[0],
        spans.e,
        e_based,
        delta_e,
        f"E is {sum(delta_e) / len(delta_e):+.3f} nats per token more likely "
        "when it is said to be based on J:",
    )
    return


@app.cell
def _(assess, lens, model, record, records):
    record_p3 = next(
        row for row in records if row["id"] == record["id"] and row["r"] is None
    )
    assessment = assess(model, lens, record_p3, "explanation")

    print("Without thinking (protocol 3), all in one call:\n")
    print("\n".join(", ".join(words) + "." for words in assessment.j))
    gap = assessment.j_word_gap
    print(
        f"\nΔ(E) = {assessment.delta:+.3f}, J-word gap = {gap:+.3f}"
        if gap is not None
        else f"\nΔ(E) = {assessment.delta:+.3f}, no J word in E"
    )
    return


@app.cell
def _(MODEL_ID, assess, case, dataclasses, json, lens, mo, model, pd, records):
    N_ITEMS = 100

    rows = [row for row in records if row["metadata"]["model_id"] == MODEL_ID]
    item_ids = list(dict.fromkeys(row["id"] for row in rows))[:N_ITEMS]
    cases = [
        (row, kind)
        for row in rows
        if row["id"] in item_ids
        for kind in (("reasoning", "explanation") if row["r"] else ("explanation",))
    ]
    assessments = []
    with mo.status.progress_bar(total=len(cases), title="Assess") as bar:
        for row, kind in cases:
            protocol = 4 if row["r"] else 3
            bar.update(
                increment=0, subtitle=f"{row['id']}: {kind}, protocol {protocol}"
            )
            assessments.append(assess(model, lens, row, kind))
            bar.update()
    # Saved so the plots below can be redrawn without the GPU.
    with open("assessments.json", "w") as f:
        json.dump([dataclasses.asdict(a) for a in assessments], f)

    domains = {row["id"]: row["domain"] for row in rows}
    results = pd.DataFrame(
        [
            {
                "id": a.id,
                "domain": domains[a.id],
                "case": case(a),
                "delta": a.delta,
                "j_word_gap": a.j_word_gap,
            }
            for a in assessments
        ]
    )

    mo.ui.table(results, label=f"{len(item_ids)} items × 3 cases assessed:")
    return assessments, cases, results


@app.cell
def _(alt, results):
    (
        alt.Chart(results, title="Δ per case, one dot per item:")
        .mark_tick()
        .encode(
            x=alt.X("delta:Q", title="Δ (nats per token)"),
            y=alt.Y("case:N", title=None),
            tooltip=["id", "domain", "delta"],
        )
    )
    return


@app.cell
def _(alt, results):
    reach = results.groupby(["case", "domain"]).delta.mean().abs().max()
    (
        alt.Chart(results, title="Δ per case and domain, mean over items:")
        .mark_rect()
        .encode(
            x=alt.X("case:N", title=None),
            y=alt.Y("domain:N", title=None),
            color=alt.Color(
                "mean(delta):Q",
                scale=alt.Scale(scheme="redblue", domain=[-reach, reach]),
                title="Δ",
            ),
            tooltip=["count()", "mean(delta)"],
        )
    )
    return


@app.cell
def _(assessments, mo, pd, spread):
    mo.ui.table(
        pd.DataFrame(spread(assessments)).T.reset_index(names="case"),
        label="Does the model read J? The spread of Δ per case:",
    )
    return


@app.cell
def _(alt, assessments, j_word_gaps, results):
    dropped = sum(g["dropped"] for g in j_word_gaps(assessments).values())
    gaps = alt.Chart(
        results.dropna(subset=["j_word_gap"]),
        title=(
            "Does the model read J? Δ on J's words minus Δ on other tokens "
            f"({dropped} transcripts without a J word in X left out):"
        ),
    ).encode(x=alt.X("j_word_gap:Q", title="J-word gap"), y=alt.Y("case:N", title=None))
    gaps.mark_tick(opacity=0.5) + gaps.mark_tick(color="crimson", thickness=3).encode(
        x="mean(j_word_gap):Q"
    )
    return


@app.cell
def _(assess, assessments, cases, dataclasses, json, lens, mismatched, mo, model):
    # The control: each X probed with the next item's J. No read-outs run.
    controls = []
    with mo.status.progress_bar(total=len(cases), title="Control") as _bar:
        for (_row, _kind), _j in zip(cases, mismatched(assessments), strict=True):
            controls.append(assess(model, lens, _row, _kind, j=_j))
            _bar.update()
    with open("assessments-mismatched.json", "w") as _f:
        json.dump([dataclasses.asdict(a) for a in controls], _f)

    mo.md(f"Scored {len(controls)} transcripts against another item's J.")
    return (controls,)


@app.cell
def _(against_control, assessments, controls, mo, pd):
    mo.ui.table(
        pd.DataFrame(against_control(assessments, controls))
        .T.round(4)
        .reset_index(names="case"),
        label="Does the model read J's content? Own J minus another item's J:",
    )
    return


@app.cell
def _(alt, assessments, case, controls, pd):
    paired = pd.DataFrame(
        [
            {
                "case": case(a),
                "id": a.id,
                "gap difference": a.j_word_gap - b.j_word_gap,
            }
            for a, b in zip(assessments, controls, strict=True)
            if a.j_word_gap is not None and b.j_word_gap is not None
        ]
    )
    pairs = alt.Chart(
        paired,
        title="J-word gap with the item's own J minus with another item's J:",
    ).encode(
        x=alt.X("gap difference:Q", title="gap difference"),
        y=alt.Y("case:N", title=None),
    )
    pairs.mark_tick(opacity=0.5) + pairs.mark_tick(color="crimson", thickness=3).encode(
        x="mean(gap difference):Q"
    )
    return


if __name__ == "__main__":
    app.run()
