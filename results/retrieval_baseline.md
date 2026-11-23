# Retrieval baseline: BM25

- Docs: `huggingface/transformers` @ `v5.24.0` (`eeee11099135`)
- Corpus: 77 chunks across 7 pages; seeds fixed; no LLM involved
- Probes: 21 hand-written retrieval probes (not the evaluation set)
- Index: BM25 k1=1.5, b=0.75, avgdl=138.38

Recall counts a hit when the retrieved chunk belongs to any gold page and the gold
section is on that chunk's heading path. Page recall is a looser diagnostic.

## Overall

| metric | @1 | @3 | @5 |
|---|---|---|---|
| section recall | 0.714 | 0.905 | 0.952 |
| page recall | 0.905 | 1.0 | 1.0 |

MRR@5 = **0.819**, median rank = 1.0, n = 21

## Per topic

| topic | n | recall@1 | recall@3 | recall@5 | MRR@5 |
|---|---|---|---|---|---|
| fine-tuning | 8 | 0.625 | 0.875 | 1.0 | 0.775 |
| generation | 7 | 1.0 | 1.0 | 1.0 | 1.0 |
| loading | 6 | 0.5 | 0.833 | 0.833 | 0.667 |

## Per probe

| qid | topic | question | gold rank | top-1 result |
|---|---|---|---|---|
| r-load-01 | loading | How do I load a pretrained model with AutoModel? | 2 | peft#loading-an-adapter |
| r-load-02 | loading | How do I load a model in a lower precision to save memory? | miss | models#large-models |
| r-load-03 | loading | What is Big Model Inference? | 1 | models#big-model-inference |
| r-load-04 | loading | How do I load a model whose code lives in a custom repository? | 1 | models#custom-models |
| r-load-05 | loading | How are sharded checkpoints handled when a model is too large? | 2 | models#large-models |
| r-load-06 | loading | How can I load weights without keeping them all in CPU memory? | 1 | models#disk-offloading |
| r-gen-01 | generation | What is the default decoding strategy for generate? | 1 | llm_tutorial#decoding-strategy |
| r-gen-02 | generation | What is beam search? | 1 | generation_strategies#beam-search |
| r-gen-03 | generation | How do I enable sampling with temperature? | 1 | generation_strategies#sampling |
| r-gen-04 | generation | How do I save a GenerationConfig and reuse it? | 1 | llm_tutorial#saving |
| r-gen-05 | generation | Why does my generated output get cut off at 20 tokens? | 1 | llm_tutorial#output-length |
| r-gen-06 | generation | How do I create a custom generation method? | 1 | generation_strategies#creating-a-custom-generation-method |
| r-gen-07 | generation | How can I make the model produce more varied text instead of the most likely token? | 1 | generation_strategies#sampling |
| r-ft-01 | fine-tuning | How do I fine-tune a model with the Trainer? | 2 | quicktour#None |
| r-ft-02 | fine-tuning | Which learning rate and batch size go in the training configuration? | 1 | training#training-configuration |
| r-ft-03 | fine-tuning | How do I load and preprocess the ELI5 dataset? | 1 | tasks_language_modeling#load-eli5-dataset |
| r-ft-04 | fine-tuning | How do I train a causal language model on ELI5? | 2 | tasks_language_modeling#load-eli5-dataset |
| r-ft-05 | fine-tuning | How do I add a LoRA adapter to a model? | 5 | peft#hotswapping-adapters |
| r-ft-06 | fine-tuning | How do I manage multiple adapters on one model? | 1 | peft#managing-multiple-adapters |
| r-ft-07 | fine-tuning | What happens when I resume PEFT training from a checkpoint? | 1 | peft#resuming-from-a-checkpoint |
| r-ft-08 | fine-tuning | How do I train only a small number of extra parameters? | 1 | peft#None |

## Misses

- **r-load-02** (loading): How do I load a model in a lower precision to save memory?
  - candidates: models#large-models (8.7965); peft#loading-an-adapter (8.3409); training#training-configuration (8.0518)

## Known weaknesses exposed here

- No stemming: `tune` does not match `tuning`, `sample` does not match `sampling`.
- Synonyms are invisible: `varied text` never reaches `sampling`.
- Section-level gold can be beaten by a related section on the same page; page recall
  is reported separately so that distinction stays visible.
