# demo.py — outline (draft)

One entry per cell: its caption, the first line of its output, with the
cell's values filled in; then what the cell does → what it shows below the
caption. A cell whose caption needs more than one sentence does too much.

0. *(no output)* — imports.

## Record

1. "Consider a reasoning trace generated with {model_id} (cached at {dataset}@{sha}):" — set model id, dataset, sha and item id; load the item's protocol-4 row → Q, R, D with `d_probs`, E.
2. *(no output)* — the span-strip helper, visible: one column per token, colored by span, labels below; reused in 11 and 13.
3. "Where faithfulness is tested: R against J before R, E against J before D." — the span strip of this record, with J's two read points marked.
   - Idea: instead, a graph of the generation process, filled with truncated material from the record: [[Q → J] → R → J → D] → J → E, with arrows "faithful?" from R to the inner group and from E to the outer group.

## Model and lens

4. "Load {model_id} ({n_layers} layers) with the Jacobian lens {lens_file}." — `load(model_id)` → model and lens.

## J for reasoning (Verbalizing latent thoughts)

5. "The transcript as the model reads it when acting as **advisor**:" — rebuild the raw string and parse it with the row's format → the transcript with Q, R, D and E marked, all heavily truncated, the focus on the relevant parts.
6. "The lens's top words at each word of Q, layers 8–21:" — read out with jlens over Q and the generation prompt → a word × layer grid (or `compute_slice` as a picture).
7. "The transcript as the model reads it when acting as **copy typist**:" — rebuild the raw string and parse it with the row's format → the transcript with Q, R, D and E marked, all heavily truncated, the focus on the relevant parts.
8. "The same text, read to retype it rather than decide it:" — read out under the typist prompt with jlens → the copy baseline's rows.
9. "What deciding adds over retyping, as J:" — TF × IDF, filter, contrast with the baseline → the fenced `<latentthought>` block.

## Probe and score: reasoning

9. "The two contexts R is scored in; they differ only in NOT:" — build the two probed contexts → the user turn with note and J, IS next to NOT.
10. "R is {Δ:+.3f} nats per token more likely when it is said to be based on J." — teacher-force R under both arms → Δ(R).
11. "Where in R the fit sits, sentence by sentence:" — smooth Δₜ per sentence → R as a strip shaded by Δ, J's words underlined.

## Explanation, same item (quicker)

12. "J right before the decision, read over Q and R:" — read out up to D and verbalize → J for the explanation, protocol 4.
13. "E is {Δ:+.3f} nats per token more likely when it is said to be based on J:" — probe and score E → E as a strip shaded by Δ.
14. "Without thinking (protocol 3), all in one call:" — `assess(model, lens, row_p3, "explanation")` → the Assessment; one call does what cells 6–13 did step by step.

## All items

15. "{n} items × 3 cases assessed:" — `assess` over the items, for reasoning and explanation in protocols 3 and 4 → a table with one row per Assessment.
16. "Δ per case, one dot per item:" — strip plot.
17. "Δ per case and domain:" — grid of strip plots or a heatmap of means.
18. "Does the model read J? The spread of Δ per case:" — Δ's distribution per case next to a tight cluster around c.
19. "Does the model read J? Δ on J's words minus Δ on other tokens:" — the gap per item and its mean per case; {k} items drop out with no J word in X.

## Open choices

- Cell 4: the span strip or the generation graph.
- Cell 6: the `compute_slice` picture or a plain grid.
- Explanation: step by step (12–13), or straight to `assess`.
- A cell comparing J with the item's DailyDilemmas values after cell 8: a check on J, not part of the demo.
