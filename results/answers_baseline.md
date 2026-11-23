# Grounded answer baseline (untuned)

- Docs: `huggingface/transformers` @ `v5.24.0` (`eeee11099135`)
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (4bit), greedy decoding, max_new_tokens=350
- Split: **test**, n=43, k=5 passages
- Contract: `Answer:` plus inline `[n]` citations, canonical `Sources:` rendered by the harness, or exactly `Insufficient evidence in the retrieved documentation.`
- Automated only: format, abstention, citation validity. Support/factual correctness: see the label sheet.

## Overall

| metric | value |
|---|---|
| n | 43 |
| behaviour accuracy | 0.116 |
| answered | 0.023 |
| grounded answer (answerable) | 0.0 |
| citation hits a gold passage (answerable) | 0.0 |
| citation hits a gold page (answerable) | 0.0 |
| mean citation precision (answered) | 0.0 |
| model abstains on negative questions | 0.267 |
| delivered abstention on negative questions | 1.0 |
| delivered answer on negative questions | 0.0 |
| parser saves on negative questions | 0.733 |
| false abstain (answerable) | 0.964 |
| exact format | 0.093 |
| section retrieval hit@k (answerable) | 0.964 |
| citation repairs | 2 |
| forced abstentions | 38 |
| median latency (s) | 3.67 |

## By category

| category | n | behaviour | answered | grounded | format | citation precision |
|---|---|---|---|---|---|---|
| answerable | 28 | 0.036 | 0.036 | 0.0 | 0.0 | 0.0 |
| parametric | 5 | 0.6 | 0.0 | None | 0.6 | None |
| unanswerable | 10 | 0.1 | 0.0 | None | 0.1 | None |

## By topic

| topic | n | behaviour | answered | citation hit | format |
|---|---|---|---|---|---|
| fine-tuning | 18 | 0.056 | 0.056 | 0.0 | 0.0 |
| generation | 10 | 0.0 | 0.0 | 0.0 | 0.0 |
| out-of-scope | 15 | 0.267 | 0.0 | None | 0.267 |

## Over-answered negative questions

None.

## Parser saves on negative questions (model tried to answer)

- **pt-03** (parametric): unverified_url_in_answer; citations=[]; extracted: An *architecture* refers to the model's skeleton and a *checkpoint* refers to the model's weights for a given architecture. For example, [BERT](./model_doc/bert) is an architecture while [google-bert/
- **pt-05** (parametric): no_valid_citations; citations=[]; extracted: Adaptive Low-Rank Approximation.
- **un-01** (unanswerable): no_valid_citations; citations=[]; extracted: ZeRO-3 works seamlessly with Trainer, but there are some adjustments needed. Specifically, you should pass `exclude_frozen_parameters=True` when saving checkpoints with a PEFT model. This ensures that
- **un-02** (unanswerable): no_valid_citations; citations=[]; extracted: First, ensure that the necessary libraries are installed. Install `bitsandbytes` if needed:

```bash
!pip install -U transformers bitsandbytes
```

Next, prepare your model and tokenizer. Here’s how y
- **un-03** (unanswerable): no_valid_citations; citations=[]; extracted: To train a new tokenizer from scratch, follow these steps:

1. Load a dataset containing the text you want to tokenize.
2. Define how you want to tokenize the text using the `tokenizer.tokenize()` met
- **un-04** (unanswerable): no_valid_citations; citations=[]; extracted: To quantize a model with GPTQ, follow these steps:

1. Install the necessary libraries:
   ```bash
   !pip install -U transformers bitsandbytes
   ```

2. Load the model and specify the quantization c
- **un-06** (unanswerable): no_valid_citations; citations=[]; extracted: To write a custom Trainer callback, follow these steps:

1. Read the [Trainer callbacks](./trainer_callbacks) guide to understand how to hook into training events for logging, early stopping, and othe
- **un-07** (unanswerable): no_valid_citations; citations=[]; extracted: Use the `map` function within the `load_dataset` method to tokenize the dataset during streaming. Here’s how you can achieve this:

```python
from datasets import load_dataset

# Load your dataset
dat
- **un-08** (unanswerable): no_valid_citations; citations=[]; extracted: To compute BLEU for generated text, you need to follow these steps:

1. Tokenize the reference translations using the same tokenizer used during training.
2. Generate candidate translations from your 
- **un-09** (unanswerable): no_valid_citations; citations=[]; extracted: To launch a multi-GPU training run with accelerate config, follow these steps:

1. Install necessary packages:
   ```bash
   !pip install -U transformers bitsandbytes
   ```

2. Load a model with acce
- **un-10** (unanswerable): no_valid_citations; citations=[]; extracted: [max_shard_size] in save_pretrained has a default value of "50GB".

## False abstentions on answerable questions

- **gs-01** (generation): Which decoding strategy does generate use by default, and how many new tokens can it produce at most when no GenerationConfig is specified?
  - reason: no_valid_citations, retrieved: llm_tutorial#decoding-strategy
- **gs-02** (generation): Why does greedy search break down when generating long sequences?
  - reason: no_valid_citations, retrieved: generation_strategies#greedy-search
- **gs-03** (generation): How do I enable multinomial sampling instead of greedy decoding?
  - reason: no_valid_citations, retrieved: llm_tutorial#decoding-strategy
- **gs-04** (generation): What are the benefits of sampling compared with greedy search?
  - reason: no_valid_citations, retrieved: llm_tutorial#decoding-strategy
- **gs-05** (generation): How does beam search differ from greedy search?
  - reason: no_valid_citations, retrieved: generation_strategies#beam-search
- **gs-06** (generation): For which kinds of tasks is beam search best suited?
  - reason: no_valid_citations, retrieved: generation_strategies#beam-search
- **gs-07** (generation): Which files must a model repository contain to publish a custom generation method?
  - reason: no_valid_citations, retrieved: generation_strategies#creating-a-custom-generation-method
- **gs-08** (generation): In a custom generation method, what must the generate function's first argument be?
  - reason: no_valid_citations, retrieved: generation_strategies#creating-a-custom-generation-method
- **gs-09** (generation): How can I find custom generation methods on the Hub?
  - reason: unverified_url_in_answer, retrieved: generation_strategies#finding-custom-generation-methods
- **gs-10** (generation): Can a custom decoding loop reuse generate's input preparation, and what does that preserve?
  - reason: no_valid_citations, retrieved: generation_strategies#reusing-generates-input-preparation
- **lm-01** (fine-tuning): What is causal language modeling?
  - reason: no_valid_citations, retrieved: tasks_language_modeling#None
- **lm-03** (fine-tuning): Which dataset and how many examples does the guide load for fine-tuning?
  - reason: no_valid_citations, retrieved: training#None
- **lm-04** (fine-tuning): How does the guide split the loaded dataset into train and test sets?
  - reason: no_valid_citations, retrieved: tasks_language_modeling#load-eli5-dataset
- **lm-05** (fine-tuning): Why does the guide flatten the ELI5 dataset before tokenizing?
  - reason: no_valid_citations, retrieved: tasks_language_modeling#load-eli5-dataset
- **lm-06** (fine-tuning): What does the group_texts function do?
  - reason: no_valid_citations, retrieved: tasks_language_modeling#preprocess
- **lm-07** (fine-tuning): Which data collator does the guide use, and with what mlm setting?
  - reason: no_valid_citations, retrieved: tasks_language_modeling#preprocess
- **lm-08** (fine-tuning): How does the guide run inference after fine-tuning the causal language model?
  - reason: no_valid_citations, retrieved: training#None
- **peft-01** (fine-tuning): Why do PEFT methods use less memory than full fine-tuning?
  - reason: no_valid_citations, retrieved: peft#next-steps
- **peft-02** (fine-tuning): Which mixin class gives all PreTrainedModel classes adapter support?
  - reason: no_valid_citations, retrieved: models#model-classes
- **peft-03** (fine-tuning): How do I attach a LoRA adapter to a model?
  - reason: no_valid_citations, retrieved: peft#hotswapping-adapters
- **peft-04** (fine-tuning): How can I fully fine-tune the language model head alongside a LoRA adapter?
  - reason: no_valid_citations, retrieved: peft#fully-fine-tuning-specific-layers
- **peft-05** (fine-tuning): Do I need to specify target_modules for Llama or Gemma models?
  - reason: no_valid_citations, retrieved: peft#choosing-which-layers-to-adapt
- **peft-06** (fine-tuning): What do Trainer checkpoints contain when training a PEFT model?
  - reason: no_valid_citations, retrieved: peft#training
- **peft-07** (fine-tuning): Does Trainer detect adapter checkpoints automatically when resuming?
  - reason: no_valid_citations, retrieved: peft#resuming-from-a-checkpoint
- **peft-08** (fine-tuning): What must a repository or directory contain for from_pretrained to load an adapter automatically?
  - reason: no_valid_citations, retrieved: peft#loading-an-adapter
- **peft-09** (fine-tuning): How do I switch to a different adapter when a model has several attached?
  - reason: no_valid_citations, retrieved: peft#managing-multiple-adapters
- **peft-10** (fine-tuning): What is adapter hotswapping and which adapter type does it support?
  - reason: no_valid_citations, retrieved: peft#hotswapping-adapters

## Answered but no cited gold passage

- **lm-02** (fine-tuning): cited [1], gold tasks_language_modeling#

## Format issues

- **gs-01**: no_valid_citations; repairs=[]; invalid=[]
- **gs-02**: no_valid_citations; repairs=[]; invalid=[]
- **gs-03**: no_valid_citations; repairs=[]; invalid=[]
- **gs-04**: no_valid_citations; repairs=[]; invalid=[]
- **gs-05**: no_valid_citations; repairs=[]; invalid=[]
- **gs-06**: no_valid_citations; repairs=[]; invalid=[]
- **gs-07**: no_valid_citations; repairs=[]; invalid=[]
- **gs-08**: no_valid_citations; repairs=[]; invalid=[]
- **gs-09**: unverified_url_in_answer; repairs=[]; invalid=[]
- **gs-10**: no_valid_citations; repairs=[]; invalid=[]
- **lm-01**: no_valid_citations; repairs=[]; invalid=[]
- **lm-02**: format; repairs=[]; invalid=[]
- **lm-03**: no_valid_citations; repairs=[]; invalid=[]
- **lm-04**: no_valid_citations; repairs=[]; invalid=[]
- **lm-05**: no_valid_citations; repairs=[]; invalid=[]
- **lm-06**: no_valid_citations; repairs=[]; invalid=[]
- **lm-07**: no_valid_citations; repairs=[]; invalid=[]
- **lm-08**: no_valid_citations; repairs=[]; invalid=[]
- **peft-01**: no_valid_citations; repairs=[]; invalid=[]
- **peft-02**: no_valid_citations; repairs=[]; invalid=[]
- **peft-03**: no_valid_citations; repairs=[]; invalid=[]
- **peft-04**: no_valid_citations; repairs=[]; invalid=[]
- **peft-05**: no_valid_citations; repairs=[]; invalid=[]
- **peft-06**: no_valid_citations; repairs=[]; invalid=[]
- **peft-07**: no_valid_citations; repairs=[]; invalid=[]
- **peft-08**: no_valid_citations; repairs=[]; invalid=[]
- **peft-09**: no_valid_citations; repairs=[]; invalid=[]
- **peft-10**: no_valid_citations; repairs=[]; invalid=[]
- **pt-03**: unverified_url_in_answer; repairs=[]; invalid=[]
- **pt-05**: no_valid_citations; repairs=[]; invalid=[]
- **un-01**: no_valid_citations; repairs=[]; invalid=[]
- **un-02**: no_valid_citations; repairs=[{'action': 'removed_out_of_range_citation', 'n': 0}]; invalid=[]
- **un-03**: no_valid_citations; repairs=[]; invalid=[]
- **un-04**: no_valid_citations; repairs=[]; invalid=[]
- **un-06**: no_valid_citations; repairs=[]; invalid=[]
- **un-07**: no_valid_citations; repairs=[]; invalid=[]
- **un-08**: no_valid_citations; repairs=[{'action': 'removed_out_of_range_citation', 'n': 0}]; invalid=[]
- **un-09**: no_valid_citations; repairs=[]; invalid=[]
- **un-10**: no_valid_citations; repairs=[]; invalid=[]

## Method notes

- Retrieval is the untouched BM25 baseline; this run isolates generation behaviour.
- A forced abstention means the parser refused the output (no valid citation, an invented
  URL, or a generation error), not that the model chose to abstain.
- Any model-written `Sources:` block or URL is discarded; the final source lines are
  rendered from the retrieved metadata, so the model cannot introduce a URL.
- These numbers are a floor for the untuned model, not a RAG quality claim.
