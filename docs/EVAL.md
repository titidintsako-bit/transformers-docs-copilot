# Evaluation set

55 hand-written questions over the pinned `v5.24.0` corpus. The set exists to answer
three separate questions, and the scoring sheet keeps them separate:

1. Did retrieval find the right section? (automatic, in `src/eval_retrieval.py`)
2. Is the generated answer supported by the cited passages? (human rubric below)
3. Did the assistant abstain when the corpus does not contain the answer? (automatic + human)

## Split rule

| Split | Answerable questions | Source pages | Purpose |
|---|---|---|---|
| `dev` | 12 | 4 SFT pages (quicktour, models, llm_tutorial, training) | prompt and pipeline development |
| `test` | 43 (28 answerable + 15 abstain) | 3 held-out pages (generation_strategies, tasks_language_modeling, peft) | headline results |

All 7 pages stay in the retrieval index. Held-out means **never used to synthesize
fine-tuning examples**, not "invisible to retrieval" - otherwise the questions would be
unanswerable by construction. The split is enforced by `src/eval_questions.py` and by an
SFT leak check later.

## Composition

| Category | Count | What it tests |
|---|---|---|
| answerable | 40 | grounded answering and citation on the gold section |
| unanswerable | 10 | abstention when the topic is outside the 7 pages |
| parametric | 5 | abstention when the base model probably knows the answer from pretraining anyway (Attention paper, BERT, GPT-2 XL, LoRA) |

Per topic: fine-tuning 21, generation 13, loading 4, overview 2, out-of-scope 15.
Per held-out page: generation_strategies 10, peft 10, tasks_language_modeling 8.

### Near-miss abstentions

Three out-of-scope questions deliberately retrieve closely related text and must still
abstain because the *specific* asked fact is absent:

- `un-01` DeepSpeed ZeRO-3 configuration - `peft#distributed-training` mentions ZeRO-3
  checkpoint behaviour, not how to configure it.
- `un-05` PEFT library license - the peft page describes the library, not its license.
- `un-10` default `max_shard_size` - the page gives the 50 GB automatic-sharding
  threshold, not the argument default.

A good abstention here is "the docs do not state this", not a guessed number.

## Gold labels

Each answerable question has `gold` entries of `{page_id, anchor}`. `anchor: null` means
"any section on this page". Anchors are validated against the chunk map, so a typo fails
`make validate-eval` instead of silently scoring zero.

Each question also has a one-sentence `reference`, visible only in the label sheet. It is
for the annotator, **not** for the model; the model sees only the question and retrieved
passages.

## Scoring rubric (fill the label sheet)

Score by comparing the answer **to the passages it cites**, not to the reference alone.

| Column | Values | Rule |
|---|---|---|
| `abstained` | 0/1 | 1 if the answer is the abstention string (or an unambiguous refusal) |
| `supported_0_1_2` | 0/1/2 | 2 = every claim supported by a cited passage; 1 = mostly supported, one minor unsupported detail; 0 = a core claim is unsupported or contradicted |
| `citation_precision_0_1` | 0/1 | 1 if every cited source is relevant to the answer; 0 if any citation is decorative or wrong |
| `citation_recall_0_1` | 0/1 | 1 if the gold section is among the citations (page + section); 0 otherwise |
| `retrieval_hit` | 0/1 | automatic: gold section appears in the top-5 retrieved chunks |

Abstention cases are scored on `abstained` only; leave the support/citation columns blank.

Expected behaviour: answerable -> answer with 1-4 sources and no unsupported claims;
unanswerable/parametric -> abstain. Answering a parametric trap counts as an
over-answer even if the fact is correct, because the corpus does not support it.

## Workflow

```bash
make validate-eval     # quotas, gold anchors, split separation
make label-sheet       # data/eval/label_sheet.csv
```

After generation runs, `src/eval_answers.py` writes a prefilled copy
(`results/label_sheet_baseline.csv`) with `retrieved_top1`, `retrieval_hit`, `abstained`,
`status`, `auto_citation_hit`, `format_ok`, `latency_s` and `repairs`. Fill
`supported_0_1_2`, `citation_precision_0_1` and `citation_recall_0_1` while looking at the
cited passages, ideally before reading the `reference` column. 55 rows is roughly 20-30
minutes. Automatic abstention is the delivered output, including parser-forced abstentions,
so use `status` to tell a model choice from a parser save.

## Headline reporting

Report on the `test` split only (n=43), overall and per topic, and break out the 28
answerable vs the 15 abstain questions. Always include `n`; a percentage over 4 loading
questions is not a trend. The dev split is for iteration and is reported separately if at
all.

## Caveats

- Single annotator, small n. Differences under ~0.08 on n=43 are noise.
- `reference` is a fact checklist, not a gold answer; wording may differ.
- Judging "supported" against cited passages is stricter than judging against the whole
  corpus, which is intentional.
- The questions are version-specific. Reusing them against another docs tag invalidates
  the gold labels.
