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
    from datetime import datetime, timezone
    from string import Template

    import datasets
    import marimo as mo
    import torch
    import transformers
    from huggingface_hub import DatasetCard, HfApi

    from j_faithfulness.formats import FORMATS
    from j_faithfulness.items import (
        load_daily_dilemmas,
        row_transcript,
        split,
        to_row,
    )

    return (
        DatasetCard,
        FORMATS,
        HfApi,
        Template,
        datasets,
        datetime,
        load_daily_dilemmas,
        mo,
        os,
        row_transcript,
        split,
        to_row,
        torch,
        timezone,
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
def _(config):
    # The model card's settings, by model id: thinking mode for R, non-thinking
    # mode for E, which follows an empty think block. The repo ships no
    # generation_config.json, so transformers' generic defaults would apply
    # otherwise. HF generate has no presence_penalty (the card: 1.5 for both).
    QWEN35_SAMPLING = {
        "r": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
        "e": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
    }
    # The 4B and 27B cards give the same settings.
    SAMPLING = {
        "Qwen/Qwen3.5-4B": QWEN35_SAMPLING,
        "Qwen/Qwen3.5-27B": QWEN35_SAMPLING,
    }
    # Recorded in each row's metadata.
    sampling = SAMPLING[config["model_id"]]
    sampling
    return (sampling,)


@app.cell
def _(hf_model, mo, tokenizer, torch, transformers):
    # Sized for the 27B on molab's 96 GB, beside its 50 GB of weights. R's
    # prompts are short, so its batch holds only a KV cache of ~64 KB per token
    # (16 full-attention layers): ~17 GB for 32 rows of 8,000 tokens.
    BATCH_SIZE = 32
    # A prompt that holds R runs to ~8,500 tokens, and its prefill peaks at
    # ~1 GB per 1,000 tokens in the MLP alone: 32 such rows ran out of memory.
    R_PROMPT_BATCH_SIZE = 16
    # E is short, so its cap only guards against a runaway reply.
    E_CAP = 2000

    think_end = tokenizer.convert_tokens_to_ids("</think>")
    eos = hf_model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    # Qwen3.5's config names only <|endoftext|>; a turn ends at <|im_end|>, the
    # tokenizer's EOS. Without it, E would run on past the end of its turn.
    eos = list(dict.fromkeys([tokenizer.eos_token_id, *eos]))

    def left_padded(prompts):
        """One batch padded on the left, so each row continues exactly where
        its raw string ends."""
        return tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(hf_model.device)

    class Progress(transformers.StoppingCriteria):
        """Never stops generation; moves the bar on as rows reach a stop
        token. A bar over batches alone would barely move: a batch holds dozens
        of items."""

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
    def sample(prompts, stops, max_new_tokens, settings, title, batch_size):
        """A continuation of each prompt, sampled with ``settings``, cut before
        its first stop token, with the stop token it ended at (None: it hit the
        cap)."""
        out = []
        starts = range(0, len(prompts), batch_size)
        with mo.status.progress_bar(total=len(prompts), title=title) as bar:
            for b, i in enumerate(starts, 1):
                enc = left_padded(prompts[i : i + batch_size])
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
                    **settings,
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
    def option_probs(prompts, options, batch_size):
        """P(D = o) after each prompt, for each of its single-token options o
        (``options[k]`` belongs to ``prompts[k]``)."""
        out = []
        for i in range(0, len(prompts), batch_size):
            enc = left_padded(prompts[i : i + batch_size])
            # No KV cache: D is one forward pass, and the cache of 16 rows that
            # hold R would take ~9 GB.
            logits = hf_model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1]
            for p, opts in zip(
                torch.softmax(logits.float(), -1),
                options[i : i + batch_size],
                strict=True,
            ):
                ids = tokenizer.convert_tokens_to_ids(opts)
                out.append(dict(zip(opts, p[ids].tolist(), strict=True)))
        return out

    return (
        BATCH_SIZE,
        E_CAP,
        R_PROMPT_BATCH_SIZE,
        eos,
        option_probs,
        sample,
        think_end,
    )


@app.cell
def _(
    BATCH_SIZE,
    config,
    eos,
    items,
    sample,
    sampling,
    think_end,
    transcript_format,
    transformers,
):
    transformers.set_seed(config["seed"])
    r_generated = sample(
        [transcript_format.reasoning_prompt(i.q) for i in items],
        [think_end, *eos],
        config["r_cap"],
        sampling["r"],
        "R",
        BATCH_SIZE,
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
def _(
    BATCH_SIZE,
    R_PROMPT_BATCH_SIZE,
    items,
    option_probs,
    reasonings,
    transcript_format,
):
    # D needs no sampling: the most likely option letter.
    def decide(prompts, batch_size):
        probs = option_probs(prompts, [i.options for i in items], batch_size)
        return [(max(p, key=p.get), p) for p in probs]

    decisions_3 = decide(
        [transcript_format.decision_prompt(i.q) for i in items], BATCH_SIZE
    )
    decisions_4 = decide(
        [
            transcript_format.decision_prompt(i.q, r, capped)
            for i, (r, capped) in zip(items, reasonings, strict=True)
        ],
        R_PROMPT_BATCH_SIZE,
    )
    [
        (i.id, d3, d4)
        for i, (d3, _), (d4, _) in zip(items, decisions_3, decisions_4, strict=True)
    ]
    return decisions_3, decisions_4


@app.cell
def _(
    BATCH_SIZE,
    E_CAP,
    R_PROMPT_BATCH_SIZE,
    decisions_3,
    decisions_4,
    eos,
    items,
    reasonings,
    sample,
    sampling,
    transcript_format,
):
    explanations_3 = sample(
        [
            transcript_format.explanation_prompt(i.q, d)
            for i, (d, _) in zip(items, decisions_3, strict=True)
        ],
        eos,
        E_CAP,
        sampling["e"],
        "E, protocol 3",
        BATCH_SIZE,
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
        sampling["e"],
        "E, protocol 4",
        R_PROMPT_BATCH_SIZE,
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
    datetime,
    decisions_3,
    decisions_4,
    explanations_3,
    explanations_4,
    items,
    model_revision,
    reasonings,
    row_transcript,
    sampling,
    timezone,
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
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
def _(Template):
    # The dataset card's text; $models and $repo_id are filled in per run.
    CARD = Template("""
# j-faithfulness transcripts

Transcripts of models deciding forced-choice questions, for research on
the faithfulness of model rationales with
[j-faithfulness](https://github.com/logikon-ai/j-faithfulness).

Models: $models.

Each item has two rows per model. In the first, thinking is off: the question, the
decision D, then "Explain." and the explanation E. In the second, thinking is
on, with the reasoning R before D.

| Field | Meaning |
|---|---|
| `id`, `q`, `options`, `domain` | the item; `options` is a list of `{label, action}` |
| `source`, `source_revision` | the dataset the item comes from |
| `r` | the text between the think tags; null with thinking off |
| `d`, `d_probs` | the decision and P(D = option), in option order |
| `e` | the explanation |
| `metadata` | model and commit, format, sampling, seed, caps, generation time |

Each run adds its rows in a new commit and replaces the rows of earlier runs
with the same model, item and thinking mode. Pin a revision and filter by
model:

```python
from datasets import load_dataset
from j_faithfulness.items import row_transcript  # pip install git+https://github.com/logikon-ai/j-faithfulness

rows = load_dataset("$repo_id", split="train", revision="<commit>")
rows = rows.filter(lambda row: row["metadata"]["model_id"] == "<model id>")
row_transcript(rows[0])  # the raw string the model saw
```

## License

For research use. The licenses of the source datasets and models named in
each row apply.
""")
    return (CARD,)


@app.cell
def _(HfApi, config, datasets, hf_token, rows):
    # The latest commit holds every earlier run's rows, so it is the base.
    # Two runs pushing at once would lose one run's rows: last push wins.
    def row_key(row):
        return (
            row["metadata"]["model_id"],
            row["source"],
            row["id"],
            row["r"] is None,
        )

    if HfApi(token=hf_token).repo_exists(
        config["repo_id"], repo_type="dataset", token=hf_token
    ):
        old_rows = datasets.load_dataset(
            config["repo_id"], split="train", token=hf_token
        ).to_list()
    else:
        old_rows = []
    new_keys = {row_key(row) for row in rows}
    kept_rows = [row for row in old_rows if row_key(row) not in new_keys]
    merged_rows = kept_rows + rows
    {
        "earlier": len(old_rows),
        "replaced": len(old_rows) - len(kept_rows),
        "new": len(rows),
        "total": len(merged_rows),
    }
    return (merged_rows,)


@app.cell
def _(CARD, DatasetCard, config, datasets, hf_token, merged_rows, rows):
    # Rows from before generated_at existed get null there: Arrow merges the
    # metadata structs.
    datasets.Dataset.from_list(merged_rows).push_to_hub(
        config["repo_id"],
        token=hf_token,
        private=True,
        commit_message=f"{len(rows) // 2} dev items from {config['model_id']}",
    )
    # The card goes on top of the README push_to_hub wrote, keeping its YAML:
    # the schema there feeds the dataset viewer.
    card = DatasetCard.load(config["repo_id"], token=hf_token)
    card.data.license = "other"
    card.data.language = ["en"]
    card.data.pretty_name = "j-faithfulness transcripts"
    card.data.tags = ["faithfulness", "interpretability", "chain-of-thought"]
    models = sorted({row["metadata"]["model_id"] for row in merged_rows})
    card.text = CARD.substitute(models=", ".join(models), repo_id=config["repo_id"])
    commit = card.push_to_hub(
        config["repo_id"], token=hf_token, commit_message="Dataset card"
    )
    # Consumers load the dataset at this revision: it holds the data and the card.
    commit.oid
    return


if __name__ == "__main__":
    app.run()
