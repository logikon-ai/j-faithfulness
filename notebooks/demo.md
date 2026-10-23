# demo.py — outline (draft)

One entry per cell: its caption, the first line of its output, with the
cell's values filled in; then what the cell does → what it shows below the
caption. A cell whose caption needs more than one sentence does too much.

0. *(no output)* — imports.

## Record

1. "Consider a reasoning trace generated with {model_id} (cached at {dataset}@{sha}):" — set model id, dataset, sha and item index; load the item's protocol-4 row → Q, R, D with `d_probs`, E.
2. *(no output)* — the span-strip helper: one column per token, colored by a field, labelled rules at given points; reused in 13 and 15.
3. "Where faithfulness is tested: R against J before R, E against J before D." — the span strip of this record, with J's two read points marked.

## Model and lens

4. "Load {model_id} ({n_layers} layers) with the Jacobian lens {lens_file}." — `load(model_id)` → model and lens.

## J for reasoning (Verbalizing latent thoughts)

5. "The transcript as the model reads it when acting as copy typist:" — the reasoning prompt under the typist role.
6. *(no output)* — the grid helper: words × layers, the best word-like token per cell; with a baseline, words it lacks in red.
7. "The lens's top words at each word of Q, layers 8–21:" — read out under the typist role → the grid.
8. "The transcript as the model reads it when acting as advisor:" — the reasoning prompt.
9. "The same text, read to decide it; in red, words the typist's read-out lacks:" — read out under the advisor role → the grid against the typist's.
10. "What deciding adds over retyping, as J:" — TF × IDF, filter, subtract the baseline → the fenced `<latentthought>` block.

## Probe and score: reasoning

11. "The two contexts R is scored in; they differ only in NOT:" — `probe` under IS and is NOT → the archive header once, the end of the user turn with note and J, and the is-NOT note.
12. "R is {Δ:+.3f} nats per token more likely when it is said to be based on J." — teacher-force R under both arms → Δ(R).
13. "Where in R the fit sits, sentence by sentence:" — Δₜ averaged per sentence → R as a strip shaded by Δ.

## Explanation, same item (quicker)

14. "J right before the decision, read over Q and R:" — read out up to D and verbalize → J for the explanation, protocol 4.
15. "E is {Δ:+.3f} nats per token more likely when it is said to be based on J:" — probe and score E → E as a strip shaded by Δ.
16. "Without thinking (protocol 3), all in one call:" — `assess(model, lens, row_p3, "explanation")` → the Assessment; one call does what cells 7–15 did step by step.

## All items

17. "{n} items × 3 cases assessed:" — `assess` over the items, for reasoning and explanation in protocols 3 and 4 → a table with one row per Assessment.
18. "Δ per case, one dot per item:" — strip plot.
19. "Δ per case and domain:" — grid of strip plots or a heatmap of means.
20. "Does the model read J? The spread of Δ per case:" — Δ's distribution per case next to a tight cluster around c.
21. "Does the model read J? Δ on J's words minus Δ on other tokens:" — the gap per item and its mean per case; {k} items drop out with no J word in X.

## Open choices

- Explanation: step by step (14–15), or straight to `assess`.
- A cell comparing J with the item's DailyDilemmas values after cell 10: a check on J, not part of the demo.
