# Grounded generation

The untuned RAG stage asks a pinned model to answer from five BM25-retrieved chunks and
to cite them. The model is never given gold labels, the evaluation questions, or access
to the docs beyond those five passages.

```
question -> BM25 top-5 -> numbered passages -> Qwen2.5-Instruct -> parser -> scored row
```

No tool calls, no web access, no hidden retrieval loop. Greedy decoding only.

## Output contract

The system prompt (`src/generate.py`) asks for exactly one of:

```
Answer: <2-6 sentences, every factual claim followed by [n]>
Sources:
[1] <passage title> - <exact passage URL>
[2] ...
```

or, when the passages do not answer the question:

```
Insufficient evidence in the retrieved documentation.
```

`Sources` lists only passages actually cited. The prompt tells the model to copy URLs
verbatim and never to introduce outside knowledge, API names, defaults or URLs.

## Parser and validator

`parse_response()` treats every model output as untrusted. It never repairs by inventing
text, only by substituting metadata that came from the retrieved passages.

| Situation | Result | Counted as |
|---|---|---|
| Valid `[n]` + matching `Sources` line | answered | no repair |
| `[n]` used inline but no `Sources` line | answered; canonical source added | `added_missing_source` |
| `Sources` URL wrong for that passage | canonical passage URL substituted | `replaced_bad_source_url` |
| Source URL lacks the section anchor | accepted; canonical anchored URL substituted | `url_missing_anchor` |
| `[n]` outside 1..5 | marker removed | `removed_out_of_range_citation` |
| No valid citation after repair | forced abstention | reason `no_valid_citations` |
| URL in the answer prose not in the passages | forced abstention | reason `unverified_url_in_answer` |
| Model raises during generation | forced abstention | reason `generation_error` |
| No chunks retrieved (BM25 zero score) | forced abstention | reason `no_retrieved_passages` |
| Exact abstention sentence | abstained | no repair |

Statuses are `answered`, `abstained` (the model chose to) and `forced_abstain` (the
parser refused its output). `behavior_accuracy` uses that model-level distinction: a forced
abstention is never scored as a correct abstention, even though the delivered output is
safe. The delivered/model split in the metrics table keeps both views visible.
Final answers can only contain URLs that were present in the retrieved passages.

`format_ok` is strict: `Answer:` prefix, at least one valid cited source, at least one
inline `[n]`, and zero repairs. A correct answer with a broken citation format is still
useful behaviourally but does not count as format-compliant.

The corpus is first-party, pinned documentation, so prompt-injection hardening is out of
scope for v1. The "no invented URLs" rule is enforced at parse time regardless.

## Automatic metrics

`src/eval_answers.py` writes `results/answers_<name>.md`, `.json`,
`results/predictions_<name>.jsonl` and a prefilled `label_sheet_<name>.csv`.

| Metric | Definition |
|---|---|
| `behavior_accuracy` | the model itself chose correctly: exact abstention on the 15 negatives, an answer on the 40 answerables; a forced abstention is a failure |
| `grounded_answer_rate` | answerable questions that were answered **and** cite a gold passage |
| `citation_hit_rate` | answerable questions whose cited passages include a gold page + section |
| `page_citation_hit_rate` | looser diagnostic: cited passage is on a gold page |
| `mean_citation_precision` | relevant cited sources / all cited sources, answered rows only |
| `model_abstain_rate_on_negative` | model emitted the exact abstention sentence on negative questions |
| `delivered_abstention_rate_on_negative` | final output was an abstention, whether chosen or forced |
| `over_answer_rate_on_negative` | a non-abstention was delivered for a negative question |
| `parser_save_rate_on_negative` | model attempted an answer on a negative question and the parser refused it |
| `format_rate` | strict contract compliance |
| `retrieval_hit_rate` | section retrieval hit@5 on answerable questions (BM25 baseline) |
| `repairs` / `forced_abstain` | parser interventions, by count |
| latency | mean and median seconds per question, generation only |

Not automated: whether the answer is actually supported and factually correct, and
citation recall (did it cite *every* gold section it used). Those three columns stay
blank in the generated label sheet; fill them by following [`EVAL.md`](EVAL.md).

## Running it

Local plumbing check (0.5B, CPU, no quantization):

```bash
make index
.venv/bin/python -m src.eval_answers --split test --limit 2 --model smoke \
  --out-prefix /tmp/answers_smoke --labels /tmp/label_sheet_smoke.csv
```

One question, verbose:

```bash
.venv/bin/python -m src.generate "Which decoding strategy does generate use by default?" --model smoke -k 5
```

Full untuned baseline (Colab T4, 1.5B in 4-bit): open
[`notebooks/colab.ipynb`](../notebooks/colab.ipynb). Equivalent CLI:

```bash
python -m src.eval_answers --split test --model base --quantize on \
  --out-prefix results/answers_baseline --labels results/label_sheet_baseline.csv
```

The notebook records the resolved model commit hash in the report.

## Local verification status

- The 0.5B CPU smoke run completed end to end: retrieval, prompt, generation, parsing,
  reports and label sheet were all produced. Both sampled answers were forced
  abstentions because the 0.5B model ignored the citation format (it answered correctly
  but emitted neither `[n]` nor `Sources`). The parser did what the contract says; this
  is a plumbing check, not a quality measurement.
- The 1.5B baseline was not run locally: this laptop has no CUDA GPU and the 3 GB model
  download stalled (176 MB in 30 minutes). The official baseline is produced by the
  Colab notebook and should be treated as pending until that report is committed.
