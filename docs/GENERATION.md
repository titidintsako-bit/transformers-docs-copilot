# Grounded generation

The untuned RAG stage asks a pinned model to answer from five BM25-retrieved chunks and
to cite them. The model is never given gold labels, the evaluation questions, or access
to the docs beyond those five passages.

```
question -> BM25 top-5 -> numbered passages -> Qwen2.5-Instruct -> parser -> scored row
```

No tool calls, no web access, no hidden retrieval loop. Greedy decoding only.

## Output contract

The model's job is only the answer text plus inline citations:

```
Answer: <2-6 sentences; every factual sentence ends with one or more [n]>
```

or, when the passages do not answer the question:

```
Insufficient evidence in the retrieved documentation.
```

The prompt explicitly tells the model **not** to write URLs or a `Sources:` section. The
application renders the final deliverable from the inline citation numbers and the
retrieved metadata:

```
Answer: ...
Sources:
[1] <passage title> - <canonical passage URL>
```

So the model never produces a URL at all; the only URLs in the output are the ones the
retrieval layer supplied. Any `Sources:` block the model writes anyway is parsed for
diagnostics and discarded, never shown.

## Parser and validator

`parse_response()` treats every model output as untrusted. Citations are taken **only**
from inline `[n]` markers; the source lines are always rendered from retrieved metadata.
The model cannot introduce a URL into the final output.

| Situation | Result | Counted as |
|---|---|---|
| Valid inline `[n]` | answered; canonical `Sources:` rendered | no repair |
| Inline `[n]` outside 1..5 | marker removed | `removed_out_of_range_citation` |
| No inline citation | forced abstention | reason `no_valid_citations` |
| Model-written `Sources:` block | diagnostic only; ignored for the output | no repair |
| Model-written source URL is wrong or missing | recorded in `invalid_source_entries`; ignored | no repair |
| URL in the answer prose | forced abstention | reason `unverified_url_in_answer` |
| Model raises during generation | forced abstention | reason `generation_error` |
| No chunks retrieved (BM25 zero score) | forced abstention | reason `no_retrieved_passages` |
| Exact abstention sentence | abstained | normalized to the exact string |

Statuses are `answered`, `abstained` (the model chose to) and `forced_abstain` (the
parser refused its output). `behavior_accuracy` uses that model-level distinction: a forced
abstention is never scored as a correct abstention, even though the delivered output is
safe. The delivered/model split in the metrics table keeps both views visible.

The delivered answer is normalised to start with `Answer:` and end with the rendered
`Sources:` block, so the user-facing contract is guaranteed even if the model forgot the
prefix. `format_ok` separately records whether the model itself supplied the prefix, at least
one valid inline `[n]`, zero out-of-range removals, and no URL in the prose. A missing
`Sources:` block is normal and is not a format failure. A model-written source block is
ignored.

The corpus is first-party, pinned documentation, so prompt-injection hardening is out of
scope for v1. The "no invented URLs" rule holds by construction: model text never becomes
a URL in the output.

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
| `format_rate` | `Answer:` prefix + at least one valid inline citation + no removals + no URL in prose |
| `retrieval_hit_rate` | section retrieval hit@5 on answerable questions (BM25 baseline) |
| `repairs` / `forced_abstain` | out-of-range citation removals and forced abstentions, by count |
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

## Verification status

- The 0.5B CPU smoke run completed end to end: retrieval, prompt, generation, parsing,
  reports and label sheet were all produced. Both sampled answers were forced
  abstentions because the 0.5B model ignored the citation format. The parser did what the
  contract says; this is a plumbing check, not a quality measurement.
- A Colab T4 quick run with the 1.5B model also completed end to end: public clone, pinned
  installs, corpus check, index and two questions at ~6 s each. Both questions were forced
  abstentions because the model emitted no inline `[n]`: one wrote a `Sources:` list
  without URLs, the other wrote no citations at all. That exposed a prompt-design problem,
  so the contract was changed to inline-citation-only with application-rendered sources.
  The quick run is being repeated before the full 43-question baseline.
- The full 1.5B baseline was not run locally: this laptop has no CUDA GPU and the 3 GB
  model download stalled. The official baseline comes from the Colab notebook and is
  pending until that report is committed.
