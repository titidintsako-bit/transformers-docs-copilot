# Tuned adapter — first-pass annotation

**Status: assistant first pass.** The rubric was applied literally to the 26 answerable test
rows where the model answered. This is not an independent human label; the user should
spot-check the flagged rows. The 16 abstentions and the `un-10` over-answer are scored by
the automatic report, not here.

## Headline

| view | result |
|---|---|
| answered rows scored | 26 |
| fully grounded (`supported = 2`) | 23/26 (88.5%) |
| mean support | 1.85 / 2 |
| citation precision | 25/26 (96.2%) |
| citation recall | 23/26 (88.5%) |
| answers the question **and** fully grounded | 15/26 (57.7%) |
| grounded but incomplete/off-topic | 8/26 (30.8%) |
| answers but has a support/citation problem | 3/26 (11.5%) |

The gap between 88.5% grounded and 57.7% fully answered is the main finding. The adapter
learned the citation contract; it still sometimes answers a nearby fact instead of the
asked fact.

## Two views of the 26 answers

|  | fully grounded | support/citation problem |
|---|---|---|
| answers the question | 15 | 3 |
| incomplete or off-topic | 8 | 0 |

`supported` measures whether the cited passage backs the answer's claims. It does **not**
measure whether the answer addresses the question. Do not report 88.5% as answer accuracy.

## Flagged rows

| qid | support / precision / recall | issue |
|---|---|---|
| `gs-01` | 1 / 1 / 0 | The 20-token limit is not in the cited `llm_tutorial` passage; recall miss. |
| `gs-04` | 2 / 1 / 1 | Asks for benefits; answer describes the mechanism and how to enable it. |
| `gs-07` | 2 / 1 / 1 | Asks which files; answer describes relative imports. |
| `gs-09` | 1 / 1 / 1 | The example repository is not in the cited passage. |
| `lm-01` | 2 / 1 / 1 | Does not give the next-token/left-only definition. |
| `lm-02` | 2 / 1 / 1 | Does not name DistilGPT2. |
| `lm-04` | 2 / 1 / 1 | Omits `test_size=0.2`. |
| `lm-06` | 2 / 1 / 1 | Does not explain what `group_texts` does. |
| `lm-07` | 2 / 1 / 0 | Does not name `DataCollatorForLanguageModeling`; cites the training page. |
| `peft-02` | 2 / 1 / 1 | Does not name `PeftAdapterMixin`. |
| `peft-03` | 0 / 0 / 0 | Answer is correct per the reference, but the cited `torch.compile` section does not support `LoraConfig`/`add_adapter`. |

The other 15 answered rows were both grounded and on-question.

## Automatic report vs first pass

| metric | automatic | first pass |
|---|---|---|
| exact format | 1.000 (43/43) | 1.000 (all scored rows) |
| behaviour accuracy | 0.930 (40/43) | not measured on this 26-row subset |
| grounded answer | 0.821 (23/28 answerable) | 0.885 (23/26 answered) |
| citation precision | 0.852 (automatic, gold-relative) | 0.962 (relevance to the answer) |
| citation recall | 0.821 (23/28 answerable) | 0.885 (23/26 answered) |

Automatic precision treats a citation as wrong when it is not the gold section; the first
pass asks whether the cited passage is relevant to the answer. That is why `gs-01` has
precision 1 here but automatic precision 0. Automatic grounded answer also includes the two
false abstentions in its denominator; the first pass only scores rows that answered.

## What to improve next

- The SFT targets are extractive and often copy a sentence near the retrieved chunk. Add
  question-focused and contrastive examples: the right passage but a nearby wrong fact,
  and the right fact with the wrong section.
- Add a completeness check to the target construction so "what is X?" examples always
  include the defining claim, not only surrounding context.
- Keep the format/citation recipe; it is no longer the bottleneck.
- Re-run the annotation after the next adapter; compare both grounded and on-question rates.

## Caveat

Single first-pass annotator (the project assistant), n=26 answered rows, literal support
rubric. The 11 flagged rows are the ones to check first; the 15 clean rows are lower risk.
