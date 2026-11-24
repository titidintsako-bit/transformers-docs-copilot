# QLoRA training

The adapter targets the gap exposed by the untuned baseline: the 1.5B model often answers
in fluent prose but omits the `Answer:` prefix and inline `[n]` citations, and it usually
answers out-of-scope questions instead of abstaining.

Training is intentionally narrow. It teaches the output contract and the abstention
decision; it does not try to make a 1.5B model a general documentation expert.

## Data recipe

`src/build_sft.py` builds the split from the **four SFT-role pages only**:

- `quicktour`, `models`, `llm_tutorial`, `training` (37 chunks).
- The three held-out test pages (`generation_strategies`, `tasks_language_modeling`,
  `peft`) are never used to synthesize examples, not even as irrelevant context. The
  training retriever is built over the 37 SFT chunks; evaluation still uses all 77.
- No evaluation question (dev or test) is used as a training question.

For each SFT chunk the builder generates up to three questions: one hand-written seed
question for the section plus two generic section questions. Each target is **extractive**:
one to three prose sentences are copied from the cited chunk, and every sentence gets the
passage number that contains it. Code fences, tables, admonitions, links and URLs are
stripped, so targets contain no URL and cannot trigger the parser's URL guard.

The current split:

| split | examples | answerable | abstain |
|---|---|---|---|
| train | 123 | 74 | 49 |
| val | 17 | 11 | 6 |
| total | 140 | 85 | 55 |

The 55 negatives are 40 general out-of-scope questions, five parametric-memory traps,
and ten Transformers-adjacent questions whose answers are absent from the pinned corpus
(TensorRT-LLM, FlashAttention-3, FSDP, llama.cpp, Unsloth, ...). They teach the
model to abstain even when the retrieved passages look related.

## QLoRA configuration

| setting | value |
|---|---|
| base | `Qwen/Qwen2.5-1.5B-Instruct` |
| quantization | 4-bit NF4, double quant, fp16 compute |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| target modules | all linear projections |
| learning rate | 2e-4, cosine, 3% warmup |
| epochs | 2 |
| batch | 1 per device, 8 gradient accumulation |
| max sequence length | 2560 |
| optimizer | paged AdamW 8-bit |
| precision | fp16 on T4 (bf16 when supported) |
| gradient checkpointing | on |

Only the assistant answer tokens are supervised. Prompt tokens, including the retrieved
passages, are masked with `-100`; the model is not trained to reproduce the passages.
The adapter is saved as a PEFT directory (`adapter_config.json` +
`adapter_model.safetensors`) and is not pushed to the Hub.

## Run it

On Colab (T4):

<https://colab.research.google.com/github/titidintsako-bit/transformers-docs-copilot/blob/main/notebooks/colab_train.ipynb>

The notebook clones the public repo, builds the SFT split, trains, evaluates the adapter
on the same 43-question test split, compares with `results/answers_baseline.json`, and
checkpoints the adapter and reports to Drive.

Equivalent commands in a GPU environment:

```bash
python -m src.build_sft
python -m src.train_qlora --train data/sft/train.jsonl --val data/sft/val.jsonl \
  --output-dir artifacts/adapter
python -m src.eval_answers --split test --model base --quantize on \
  --adapter artifacts/adapter --out-prefix results/answers_tuned \
  --labels results/label_sheet_tuned.csv
```

The local laptop has no NVIDIA GPU; `make sft` works locally, but `make train-qlora`
requires CUDA.

## Result

The full T4 run at `5ada516` (2 epochs, full split) produced:

| metric | baseline | tuned |
|---|---|---|
| `format_rate` | 0.093 | 1.000 |
| `grounded_answer_rate` | 0.000 | 0.821 |
| `behavior_accuracy` | 0.116 | 0.930 |
| `answered_rate` | 0.023 | 0.628 |
| `model_abstain_rate_on_negative` | 0.267 | 0.933 |
| `false_abstain_rate_on_answerable` | 0.964 | 0.071 |
| `forced_abstain` | 38 | 0 |
| median latency | 3.67 s | 5.02 s |

The adapter learned the contract and transferred it to the three held-out pages. Remaining
errors: one over-answer on `un-10` (default `max_shard_size`), two false abstentions
(`lm-05`, `lm-08`), and three answers citing a related section instead of the gold one.
The executed notebook is `notebooks/colab_train_run.ipynb`; the report is
`results/answers_tuned.md`. Human support/correctness scoring is still pending.

## What to compare

| metric | baseline | what an improvement means |
|---|---|---|
| `format_rate` | 0.093 | the model itself emits `Answer:` + valid inline citations |
| `grounded_answer_rate` | 0.000 | answered answerables with a gold-section citation |
| `behavior_accuracy` | 0.116 | model chooses answer vs abstain correctly |
| `model_abstain_rate_on_negative` | 0.267 | the model, not the parser, abstains on negatives |
| `false_abstain_rate_on_answerable` | 0.964 | fewer correct answers thrown away |

`delivered_abstention_rate_on_negative` is already 1.0 because the parser refuses unsafe
or uncited output. That is a safety property, not evidence that the model learned to
abstain; do not cite it as an improvement unless the model-level rate moves.

## Limitations

- The targets are extractive and somewhat templated. They teach format and citation
  discipline, not open-ended synthesis or factual breadth.
- 140 examples over four pages is a small SFT set; the adapter is a contract adapter, not
  a documentation expert.
- The test pages are held out from training, so an improvement is evidence of format
  transfer, not memorization of the test pages.
- No hyperparameter sweep. If format stays near zero after training, the next variable is
  a stronger base model (Qwen2.5-3B/4B), not more prompt variants.
