# Grounded answer report (tuned adapter)

- Docs: `huggingface/transformers` @ `v5.24.0` (`eeee11099135`)
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4), greedy decoding, max_new_tokens=350
- Adapter: `artifacts/adapter`
- Split: **test**, n=43, k=5 passages
- Contract: `Answer:` plus inline `[n]` citations, canonical `Sources:` rendered by the harness, or exactly `Insufficient evidence in the retrieved documentation.`
- Automated only: format, abstention, citation validity. Support/factual correctness: see the label sheet.

## Overall

| metric | value |
|---|---|
| n | 43 |
| behaviour accuracy | 0.93 |
| answered | 0.628 |
| grounded answer (answerable) | 0.821 |
| citation hits a gold passage (answerable) | 0.821 |
| citation hits a gold page (answerable) | 0.857 |
| mean citation precision (answered) | 0.852 |
| model abstains on negative questions | 0.933 |
| delivered abstention on negative questions | 0.933 |
| delivered answer on negative questions | 0.067 |
| parser saves on negative questions | 0.0 |
| false abstain (answerable) | 0.071 |
| exact format | 1.0 |
| section retrieval hit@k (answerable) | 0.964 |
| citation repairs | 0 |
| forced abstentions | 0 |
| median latency (s) | 5.36 |

## By category

| category | n | behaviour | answered | grounded | format | citation precision |
|---|---|---|---|---|---|---|
| answerable | 28 | 0.929 | 0.929 | 0.821 | 1.0 | 0.885 |
| parametric | 5 | 1.0 | 0.0 | None | 1.0 | None |
| unanswerable | 10 | 0.9 | 0.1 | None | 1.0 | 0.0 |

## By topic

| topic | n | behaviour | answered | citation hit | format |
|---|---|---|---|---|---|
| fine-tuning | 18 | 0.889 | 0.889 | 0.778 | 1.0 |
| generation | 10 | 1.0 | 1.0 | 0.9 | 1.0 |
| out-of-scope | 15 | 0.933 | 0.067 | None | 1.0 |

## Over-answered negative questions

- **un-10** (unanswerable): What is the default value of max_shard_size in save_pretrained?
  - answer: Answer: Parameters load in parallel and peak memory only depends on model size [1]. Use `max_shard_size` in `~PreTrainedModel.save_pretrained` to set the maximum checkpoint size before sharding [1].

Sources:
[1] Loading models - https://gi

## Parser saves on negative questions (model tried to answer)

None.

## False abstentions on answerable questions

- **lm-05** (fine-tuning): Why does the guide flatten the ELI5 dataset before tokenizing?
  - reason: None, retrieved: tasks_language_modeling#load-eli5-dataset
- **lm-08** (fine-tuning): How does the guide run inference after fine-tuning the causal language model?
  - reason: None, retrieved: training#None

## Answered but no cited gold passage

- **gs-01** (generation): cited [1], gold generation_strategies#greedy-search
- **lm-07** (fine-tuning): cited [2], gold tasks_language_modeling#preprocess
- **peft-03** (fine-tuning): cited [4], gold peft#add-an-adapter

## Format issues

None.

## Method notes

- Retrieval is the untouched BM25 baseline; this run isolates generation behaviour.
- A forced abstention means the parser refused the output (no valid citation, an invented
  URL, or a generation error), not that the model chose to abstain.
- Any model-written `Sources:` block or URL is discarded; the final source lines are
  rendered from the retrieved metadata, so the model cannot introduce a URL.
- These are automated contract metrics, not a RAG quality claim; support and
  factual correctness stay in the human label sheet.
