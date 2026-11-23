# Transformers Docs Copilot

Answers questions about a small, **version-pinned** slice of the official Hugging Face
Transformers documentation, and cites the section each claim comes from. Built as a
first RAG project: keyword retrieval baseline, untuned RAG, then QLoRA fine-tuning for
answer format, citation behavior and abstention.

**Status: Milestone 1 of 6 (provenance fetch).** Retrieval, evaluation, tuning and the demo
land in the next milestones; nothing here claims results yet.

## Pinned corpus

- `huggingface/transformers` @ `v5.24.0`, commit resolved at fetch time
- 7 pages, three topics: loading models, text generation, fine-tuning
- Docs are Apache-2.0; see [`docs/DATA-LICENSE.md`](docs/DATA-LICENSE.md) for attribution,
  per-page SHA-256 hashes and scope decisions
- 4 pages feed fine-tuning data, 3 pages are held out for evaluation (but stay in the index)

## Quickstart (local, no GPU)

```bash
make setup      # uv venv + editable install
make fetch      # download pinned pages, write manifest + license note
make check      # offline provenance verification
make test       # unit tests (slugger, hashes, roles, license)
```

The local machine has no NVIDIA GPU; 4-bit QLoRA training runs on a Colab T4. The local
environment is for ingestion, retrieval, evaluation and a CPU smoke test with a 0.5B model.

## Layout

```
configs/project.yaml     pinned docs, page roles/topics, model + QLoRA settings
src/text.py              slugger, heading parser, hashing
src/fetch_docs.py        provenance fetch + offline check
data/raw/                fetched pages (byte-for-byte), manifest.jsonl, docset.json
docs/DATA-LICENSE.md     generated licence/provenance note
tests/                   provenance + text-parser tests
```
