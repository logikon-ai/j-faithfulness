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
    import os

    import datasets
    import marimo as mo
    import torch
    import transformers
    from huggingface_hub import HfApi

    from j_faithfulness.formats import FORMATS
    from j_faithfulness.items import (
        load_daily_dilemmas,
        row_transcript,
        split,
        to_row,
    )

    return (
        FORMATS,
        HfApi,
        datasets,
        load_daily_dilemmas,
        mo,
        os,
        row_transcript,
        split,
        to_row,
        torch,
        transformers,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Generate transcripts

    Generates the two transcripts of each dev item (DESIGN.md, Transcripts):
    protocol 3 (thinking off) and protocol 4 (thinking on). Three batched
    passes: R in protocol 4, D in both, E in both. Each transcript becomes one
    row of a Hugging Face dataset; pin the commit printed at the end.
    """)
    return


@app.cell
def _(mo):
    settings = (
        mo.md("""
        {model_id}

        {n_items}

        {repo_id}

        {hf_token}

        {seed}

        {r_cap}
        """)
        .batch(
            model_id=mo.ui.text(
                value="Qwen/Qwen3.5-4B", label="Model id", full_width=True
            ),
            n_items=mo.ui.number(start=1, stop=680, value=50, label="Dev items"),
            repo_id=mo.ui.text(label="Dataset repo id", full_width=True),
            hf_token=mo.ui.text(
                kind="password", label="HF token (empty: HF_TOKEN)", full_width=True
            ),
            seed=mo.ui.number(value=0, label="Seed"),
            r_cap=mo.ui.number(start=1, value=8000, label="R cap (tokens)"),
        )
        # A form, so editing a field doesn't rerun hours of generation.
        .form(
            submit_button_label="Generate",
            validate=lambda v: None if v["repo_id"] else "Set the dataset repo id.",
        )
    )
    settings
    return (settings,)


@app.cell
def _(mo, settings):
    mo.stop(settings.value is None, mo.md("_Press **Generate** to start._"))
    config = dict(settings.value)
    config["hf_token"] = config["hf_token"] or None
    config["n_items"] = int(config["n_items"])
    config["seed"] = int(config["seed"])
    config["r_cap"] = int(config["r_cap"])
    {k: v for k, v in config.items() if k != "hf_token"}
    return (config,)


@app.cell
def _(config, os):
    # The env var is read only when the field is empty, so a typed token wins.
    hf_token = config["hf_token"] or os.environ.get("HF_TOKEN")
    return (hf_token,)


@app.cell
def _(FORMATS, config):
    transcript_format = FORMATS[config["model_id"]]
    transcript_format.name
    return (transcript_format,)


@app.cell
def _(config, hf_token, load_daily_dilemmas, split):
    # Always dev: the test items stay untouched (DESIGN.md, Generation and cache).
    items = split(load_daily_dilemmas(hf_token), "dev", config["n_items"])
    [(i.id, i.domain) for i in items]
    return (items,)


@app.cell
def _(HfApi, config, hf_token, torch, transformers):
    # Load at the resolved commit, so the recorded revision is the one used.
    model_revision = HfApi(token=hf_token).model_info(config["model_id"]).sha
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=model_revision,
        token=hf_token,
        dtype=torch.bfloat16,
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["model_id"], revision=model_revision, token=hf_token
    )
    model_revision
    return hf_model, model_revision, tokenizer


@app.cell
def _(hf_model):
    # R and E are sampled with the model's defaults; these are recorded per row.
    sampling = {
        k: getattr(hf_model.generation_config, k, None)
        for k in ("do_sample", "temperature", "top_p", "top_k", "min_p")
    }
    sampling
    return (sampling,)


@app.cell
def _(hf_model, mo, tokenizer, torch, transformers):
    # Generous: one batch holds every item on molab's 96 GB.
    BATCH_SIZE = 64
    # E is short, so its cap only guards against a runaway reply.
    E_CAP = 2000

    think_end = tokenizer.convert_tokens_to_ids("</think>")
    eos = hf_model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)

    def left_padded(prompts):
        """One batch padded on the left, so each row continues exactly where
        its raw string ends."""
        return tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(hf_model.device)

    class Progress(transformers.StoppingCriteria):
        """Never stops generation; moves the bar on as rows reach a stop
        token. A bar over batches alone wouldn't move: one batch holds every
        item."""

        def __init__(self, bar, stops, n_rows, batch, max_new_tokens):
            self.bar, self.batch, self.max_new_tokens = bar, batch, max_new_tokens
            self.stops = torch.tensor(stops, device=hf_model.device)
            self.done = torch.zeros(n_rows, dtype=torch.bool, device=hf_model.device)
            self.reported = self.step = 0

        def __call__(self, input_ids, scores, **kwargs):
            self.step += 1
            self.done |= torch.isin(input_ids[:, -1], self.stops)
            # Every 20 tokens: reading the done count syncs with the GPU.
            if self.step % 20 == 0:
                self.report()
            return torch.zeros_like(self.done)

        def report(self, finished=None):
            finished = int(self.done.sum()) if finished is None else finished
            self.bar.update(
                increment=finished - self.reported,
                subtitle=f"{self.batch}, token {self.step} of {self.max_new_tokens}",
            )
            self.reported = finished

    @torch.no_grad()
    def sample(prompts, stops, max_new_tokens, title):
        """A sampled continuation of each prompt, cut before its first stop
        token, with the stop token it ended at (None: it hit the cap)."""
        out = []
        starts = range(0, len(prompts), BATCH_SIZE)
        with mo.status.progress_bar(total=len(prompts), title=title) as bar:
            for b, i in enumerate(starts, 1):
                enc = left_padded(prompts[i : i + BATCH_SIZE])
                progress = Progress(
                    bar,
                    stops,
                    enc.input_ids.shape[0],
                    f"batch {b} of {len(starts)}",
                    max_new_tokens,
                )
                ids = hf_model.generate(
                    **enc,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=stops,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=transformers.StoppingCriteriaList([progress]),
                )
                # Rows that hit the cap count as finished, too.
                progress.report(finished=enc.input_ids.shape[0])
                for row in ids[:, enc.input_ids.shape[-1] :].tolist():
                    end = next((k for k, t in enumerate(row) if t in stops), None)
                    text = tokenizer.decode(row[:end], skip_special_tokens=False)
                    out.append((text, None if end is None else row[end]))
        return out

    @torch.no_grad()
    def option_probs(prompts, options):
        """P(D = o) after each prompt, for each of its single-token options o
        (``options[k]`` belongs to ``prompts[k]``)."""
        out = []
        for i in range(0, len(prompts), BATCH_SIZE):
            enc = left_padded(prompts[i : i + BATCH_SIZE])
            logits = hf_model(**enc, logits_to_keep=1).logits[:, -1]
            for p, opts in zip(
                torch.softmax(logits.float(), -1),
                options[i : i + BATCH_SIZE],
                strict=True,
            ):
                ids = tokenizer.convert_tokens_to_ids(opts)
                out.append(dict(zip(opts, p[ids].tolist(), strict=True)))
        return out

    return E_CAP, eos, option_probs, sample, think_end


@app.cell
def _(config, items, sample, think_end, eos, transformers, transcript_format):
    transformers.set_seed(config["seed"])
    r_generated = sample(
        [transcript_format.reasoning_prompt(i.q) for i in items],
        [think_end, *eos],
        config["r_cap"],
        "R",
    )
    # An R that didn't reach </think> is closed with "..." (DESIGN.md, Generation
    # and cache). The model puts a newline before </think>; the format adds it.
    reasonings = [
        (text.removesuffix("\n"), False) if stop == think_end else (text, True)
        for text, stop in r_generated
    ]
    {"capped": sum(capped for _, capped in reasonings), "of": len(reasonings)}
    return (reasonings,)


@app.cell
def _(items, option_probs, reasonings, transcript_format):
    # D needs no sampling: the most likely option letter.
    def decide(prompts):
        probs = option_probs(prompts, [i.options for i in items])
        return [(max(p, key=p.get), p) for p in probs]

    decisions_3 = decide([transcript_format.decision_prompt(i.q) for i in items])
    decisions_4 = decide(
        [
            transcript_format.decision_prompt(i.q, r, capped)
            for i, (r, capped) in zip(items, reasonings, strict=True)
        ]
    )
    [
        (i.id, d3, d4)
        for i, (d3, _), (d4, _) in zip(items, decisions_3, decisions_4, strict=True)
    ]
    return decisions_3, decisions_4


@app.cell
def _(
    E_CAP,
    decisions_3,
    decisions_4,
    eos,
    items,
    reasonings,
    sample,
    transcript_format,
):
    explanations_3 = sample(
        [
            transcript_format.explanation_prompt(i.q, d)
            for i, (d, _) in zip(items, decisions_3, strict=True)
        ],
        eos,
        E_CAP,
        "E, protocol 3",
    )
    explanations_4 = sample(
        [
            transcript_format.explanation_prompt(i.q, d, r, capped)
            for i, (d, _), (r, capped) in zip(
                items, decisions_4, reasonings, strict=True
            )
        ],
        eos,
        E_CAP,
        "E, protocol 4",
    )
    {
        "capped, protocol 3": sum(stop is None for _, stop in explanations_3),
        "capped, protocol 4": sum(stop is None for _, stop in explanations_4),
    }
    return explanations_3, explanations_4


@app.cell
def _(
    E_CAP,
    config,
    decisions_3,
    decisions_4,
    explanations_3,
    explanations_4,
    items,
    model_revision,
    reasonings,
    row_transcript,
    sampling,
    to_row,
    transcript_format,
):
    run = {
        "model_id": config["model_id"],
        "model_revision": model_revision,
        "format": transcript_format.name,
        "sampling": sampling,
        "seed": config["seed"],
        "r_cap": config["r_cap"],
        "e_cap": E_CAP,
    }
    # One row per transcript: protocol 3 (r is None), then protocol 4.
    rows = []
    for item, (r, r_capped), (d3, p3), (d4, p4), (e3, stop3), (e4, stop4) in zip(
        items,
        reasonings,
        decisions_3,
        decisions_4,
        explanations_3,
        explanations_4,
        strict=True,
    ):
        for _r, _capped, _d, _p, _e, _stop in (
            (None, None, d3, p3, e3, stop3),
            (r, r_capped, d4, p4, e4, stop4),
        ):
            _text = transcript_format.transcript(item.q, _d, _e, _r, bool(_capped))
            # The stored R is the text between the think tags, cap mark
            # included, as the model saw it.
            _spans = transcript_format.parse(_text)
            rows.append(
                {
                    **to_row(item),
                    "r": None if _r is None else _text[slice(*_spans.r)],
                    "d": _d,
                    # In option order, like the row's options.
                    "d_probs": [_p[o] for o in item.options],
                    "e": _e,
                    "metadata": {
                        **run,
                        "r_capped": _capped,
                        "e_capped": _stop is None,
                    },
                }
            )
            assert row_transcript(rows[-1]) == _text
    print(row_transcript(rows[1]))
    return (rows,)


@app.cell
def _(config, datasets, hf_token, rows):
    commit = datasets.Dataset.from_list(rows).push_to_hub(
        config["repo_id"],
        token=hf_token,
        private=True,
        commit_message=f"{len(rows) // 2} dev items from {config['model_id']}",
    )
    # Consumers load the dataset at this revision.
    commit.oid
    return


if __name__ == "__main__":
    app.run()
