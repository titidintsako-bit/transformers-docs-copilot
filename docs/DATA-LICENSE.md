# Data and reuse terms

## Source

- Repository: <https://github.com/huggingface/transformers>
- Pinned tag: `v5.24.0`
- Resolved commit: `eeee11099135592f1d03c2871fbd3d8132e5a4d7`
- Documentation root: `docs/source/en`
- Retrieved: 2026-11-23T14:53:03Z

## License

The Transformers repository, including `docs/source/en`, is licensed under the
**Apache License, Version 2.0** (https://github.com/huggingface/transformers/blob/eeee11099135592f1d03c2871fbd3d8132e5a4d7/LICENSE). Every fetched page also carries an
Apache-2.0 copyright header, which is preserved verbatim in `data/raw/md/`. No per-page
exception was found for the files below; if one exists, exclude that page and regenerate.

### Attribution

Documentation text: © The Hugging Face Team, licensed under the Apache License, Version 2.0.
This project is not affiliated with or endorsed by Hugging Face.

### Changes

The source files are stored byte-for-byte. Downstream, this project removes license
comments before indexing, splits the text into sections, and adds breadcrumbs and IDs for
retrieval. Those transformations are documented in `data/raw/docset.json` and the chunking code.

## Fetched pages

| Page ID | Title | Role | Bytes | SHA-256 |
|---|---|---|---|---|
| `quicktour` | Quickstart | sft | 12,489 | `20d63e718525` |
| `models` | Loading models | sft | 13,355 | `c8df2f1dd26c` |
| `llm_tutorial` | Text generation | sft | 15,675 | `eff6d8e41b8f` |
| `training` | Fine-tuning | sft | 8,862 | `1a12e1548b0a` |
| `generation_strategies` | Generation strategies | test | 19,697 | `cf819920c69a` |
| `tasks_language_modeling` | Causal language modeling | test | 13,922 | `6dcf5ada2e20` |
| `peft` | Parameter-efficient fine-tuning | test | 10,375 | `97144da5d061` |

`role: sft` pages may be used to synthesize fine-tuning data. `role: test` pages are held out
from all fine-tuning data but remain in the retrieval index, otherwise the evaluation would be
unanswerable by construction.

## Scope decisions

- `tasks/text_generation.md` does not exist at `v5.24.0`; `tasks/language_modeling.md`
  is used as the text-generation task page.
- `main_classes/model.md` and `main_classes/trainer.md` were excluded as stubs (under 3 KB).
- `generation_features.md`, `chat_templating.md` and `serialization.md` were excluded to keep
  three topics with minimal overlap.

## Model weights

The documentation license does not cover model weights. The base model
(`Qwen/Qwen2.5-1.5B-Instruct`) is published under Apache-2.0; the fallback
(`Qwen/Qwen3-4B-Instruct-2507`) is also Apache-2.0. Verify the model card before redistributing
any merged or fine-tuned weights.
