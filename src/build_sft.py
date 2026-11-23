"""Build synthetic SFT examples from SFT-role chunks only.

The held-out pages are never used to synthesize training data, not even as irrelevant
context. Retrieval for training therefore runs over an index of the four SFT pages
only; evaluation still uses the full seven-page index.

Targets are extractive: the answer sentences are copied from the cited chunk and each
sentence gets the passage number that contains it. This teaches the output contract
(``Answer:`` + inline citations) without asking a weaker model to invent facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from src.bm25 import tokenize
from src.common import CHUNKS_PATH, ROOT, load_config, read_jsonl, write_jsonl
from src.generate import ABSTAIN_TEXT, build_messages
from src.retrieve import Retriever

SFT_DIR = ROOT / "data" / "sft"
SEED = 42
MAX_CONTEXT_CHARS = 9000

SEED_QUESTIONS = {
    "quicktour:set-up:0": "How do I authenticate with the Hugging Face Hub?",
    "quicktour:pretrained-models:0": "Which classes should I know before loading a pretrained model?",
    "quicktour:pipeline:0": "What can I do with the Pipeline class in the quickstart?",
    "quicktour:trainer:0": "What components does Trainer need to build batches of data?",
    "quicktour:next-steps:0": "What should I learn after the quickstart?",
    "models:models-and-configurations:0": "What is the difference between an architecture and a checkpoint?",
    "models:model-classes:0": "When would I use AutoModel instead of a model-specific class?",
    "models:large-models:0": "Why do large models need extra memory when loading?",
    "models:sharded-checkpoints:0": "When does save_pretrained shard a checkpoint?",
    "models:big-model-inference:0": "What does Big Model Inference do?",
    "models:disk-offloading:0": "When should I set offload_folder?",
    "models:model-data-type:0": "How does the dtype setting affect model loading?",
    "models:custom-models:0": "What should I be careful about when loading a custom model?",
    "llm_tutorial:default-generate:0": "How do I load an LLM with less memory?",
    "llm_tutorial:generation-configuration:0": "Where are generation settings stored?",
    "llm_tutorial:saving:0": "Where should I put a custom generation configuration?",
    "llm_tutorial:common-options:0": "Which generation options are commonly used?",
    "llm_tutorial:output-length:0": "How many tokens does generate return by default?",
    "llm_tutorial:padding-side:0": "What padding side do LLM inputs need?",
    "llm_tutorial:prompt-format:0": "Why does a chat model need a chat template?",
    "llm_tutorial:decoding-strategy:0": "What does the text generation guide say about the default decoding strategy?",
    "training:tokenization:0": "How do I tokenize the training text column?",
    "training:loading-a-model:0": "How does dtype auto help when loading a model for training?",
    "training:training-configuration:0": "How do I set training duration and batch size with TrainingArguments?",
    "training:training:0": "How do I start training with Trainer?",
    "training:next-steps:0": "What Trainer features can I read about next?",
}

NEGATIVE_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote the novel Pride and Prejudice?",
    "How do I bake sourdough bread?",
    "What is the boiling point of mercury?",
    "How do I change a bicycle tire?",
    "What is the Krebs cycle?",
    "Who painted the Mona Lisa?",
    "How do I train a puppy to sit?",
    "What is the speed of light in a vacuum?",
    "What is the plot of Hamlet?",
    "How do I grow tomatoes in a greenhouse?",
    "What causes earthquakes?",
    "How do I write a sonnet?",
    "What is the population of Brazil?",
    "How do I fix a leaking kitchen tap?",
    "What is the chemical formula for table salt?",
    "How do I play the C major scale on guitar?",
    "What is the history of the Roman Empire?",
    "How do I configure a Kubernetes ingress with TLS?",
    "How do I set up a PostgreSQL primary-replica cluster?",
    "What is the best diet for a diabetic cat?",
    "How do I solve a quadratic equation?",
    "What is the difference between TCP and UDP?",
    "How do I make cold brew coffee?",
    "How do I convert miles to kilometers?",
    "What is the capital of Japan?",
    "How do I knit a scarf?",
    "What is the largest planet in the solar system?",
    "How do I replace a car battery?",
    "What is the Pythagorean theorem?",
    "How do I write a cover letter?",
    "What is the role of chlorophyll in photosynthesis?",
    "How do I set up a home Wi-Fi mesh?",
    "What is the plot of The Great Gatsby?",
    "How do I train for a marathon?",
    "What is the freezing point of water in Fahrenheit?",
    "How do I install a ceiling fan?",
    "Who composed the Ninth Symphony?",
    "What is the difference between fiction and nonfiction?",
    "How do I make a paper airplane?",
    "Who invented the telephone?",
    "What is the tallest mountain in the world?",
    "Who painted the Sistine Chapel ceiling?",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
    "How do I deploy a model with TensorRT-LLM?",
    "How do I use FlashAttention-3 in a training script?",
    "How do I configure FSDP sharding for a language model?",
    "How do I run inference with llama.cpp?",
    "How do I use Unsloth for fine-tuning?",
    "How do I convert a model to ONNX?",
    "How do I use Ray Train for distributed training?",
    "How do I log experiments with Weights & Biases?",
    "How do I use PEFT with diffusion models?",
    "How do I train a reward model for RLHF?",
]

_CODE_BLOCK = re.compile(r"```.*?```", re.S)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_TAG = re.compile(r"<[^>]+>")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.M)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.M)
_BREADCRUMB = re.compile(r"^Transformers v[\d.]+ > [^\n]+\n+")
_DOC_CITATION = re.compile(r"\[\d+\]")
_ADMONITION = re.compile(r"\[!(?:TIP|NOTE|WARNING|IMPORTANT|CAUTION)\]")
_API_REF = re.compile(r"\[`([^`]+)`\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"'])")


def clean_prose(text: str) -> str:
    """Strip markup, code, tables, links and URLs so answers contain prose only."""
    text = _BREADCRUMB.sub("", text.replace("\r", ""))
    text = _CODE_BLOCK.sub(" ", text)
    text = _ADMONITION.sub(" ", text)
    text = re.sub(r"^\s*>\s?", "", text, flags=re.M)
    text = _API_REF.sub(r"`\1`", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _TABLE_LINE.sub("", text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _LIST_MARKER.sub("", text)
    text = _DOC_CITATION.sub("", text)
    text = re.sub(r"\[\[[^\]]+\]\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text)
    result = []
    for candidate in sentences:
        sentence = candidate.strip()
        if not 40 <= len(sentence) <= 360:
            continue
        if sentence[-1] not in ".!?":
            continue
        if "http" in sentence or sentence.startswith(("def ", "class ", "import ")):
            continue
        if re.match(r"^(Refer to|Read the|See the|Check out|Visit)\b", sentence, re.I):
            continue
        result.append(sentence)
    return result


def select_sentences(text: str, question: str, max_sentences: int = 3) -> list[str]:
    sentences = split_sentences(clean_prose(text))
    if not sentences:
        return []
    query = set(tokenize(question))
    scored = []
    for index, sentence in enumerate(sentences):
        overlap = len(query & set(tokenize(sentence)))
        density = overlap / (1 + len(tokenize(sentence)) ** 0.5)
        scored.append((density, index, sentence))
    top = sorted(scored, reverse=True)[:max_sentences]
    if not any(score for score, _, _ in top):
        top = [(0.0, index, sentence) for index, sentence in enumerate(sentences[:max_sentences])]
    chosen = sorted(top, key=lambda item: item[1])
    return [sentence for _, _, sentence in chosen]


def make_target(sentences: list[str], citation: int) -> str:
    parts = []
    for sentence in sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if cleaned[-1] in ".!?":
            cleaned = cleaned[:-1]
        if not cleaned or cleaned[-1] in ":;":
            continue
        parts.append(f"{cleaned} [{citation}].")
    return "Answer: " + " ".join(parts)


def question_variants(chunk: dict) -> list[str]:
    short = chunk["section"].split(" > ")[-1]
    page = chunk["page_title"]
    questions = []
    seed = SEED_QUESTIONS.get(chunk["chunk_id"])
    if seed:
        questions.append(seed)
    questions.extend(
        [
            f"What does the documentation say about {short}?",
            f"What should I know about {short} in the {page} guide?",
        ]
    )
    return list(dict.fromkeys(questions))[:3]


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _fits(passages: list[dict], limit: int = MAX_CONTEXT_CHARS) -> bool:
    return sum(len(passage["text"]) for passage in passages) <= limit


def build_examples(
    chunks: list[dict],
    retriever: Retriever,
    *,
    negatives: list[str] | None = None,
    k: int = 5,
    max_context_chars: int = MAX_CONTEXT_CHARS,
) -> list[dict]:
    sft_chunks = [chunk for chunk in chunks if chunk["role"] == "sft"]
    examples: list[dict] = []
    skipped = {"not_retrieved": 0, "too_long": 0, "no_prose": 0}

    for chunk in sft_chunks:
        for question in question_variants(chunk):
            passages = retriever.search(question, k=k)
            gold_index = next(
                (index for index, passage in enumerate(passages) if passage["chunk_id"] == chunk["chunk_id"]),
                None,
            )
            if gold_index is None:
                skipped["not_retrieved"] += 1
                continue
            if not _fits(passages, max_context_chars):
                skipped["too_long"] += 1
                continue
            sentences = select_sentences(chunk["text"], question)
            if not sentences:
                skipped["no_prose"] += 1
                continue
            target = make_target(sentences, gold_index + 1)
            if not target.removeprefix("Answer: ").strip():
                skipped["no_prose"] += 1
                continue
            examples.append(
                {
                    "id": f"{chunk['chunk_id']}:{_short_hash(question)}",
                    "group": chunk["chunk_id"],
                    "category": "answerable",
                    "question": question,
                    "source_chunk": chunk["chunk_id"],
                    "citation": gold_index + 1,
                    "target": target,
                    "messages": build_messages(question, passages)
                    + [{"role": "assistant", "content": target}],
                }
            )

    for index, question in enumerate(negatives or NEGATIVE_QUESTIONS):
        passages = retriever.search(question, k=k)
        if not _fits(passages, max_context_chars):
            passages = passages[:3]
        examples.append(
            {
                "id": f"negative:{index:02d}",
                "group": f"negative:{index:02d}",
                "category": "abstain",
                "question": question,
                "source_chunk": None,
                "citation": None,
                "target": ABSTAIN_TEXT,
                "messages": build_messages(question, passages)
                + [{"role": "assistant", "content": ABSTAIN_TEXT}],
            }
        )

    if skipped:
        print("build_sft skips:", json.dumps(skipped, sort_keys=True), file=sys.stderr)
    return examples


def split_examples(
    examples: list[dict], *, val_fraction: float = 0.1, seed: int = SEED
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    positive = sorted({row["group"] for row in examples if row["category"] == "answerable"})
    negative = sorted({row["group"] for row in examples if row["category"] == "abstain"})
    rng.shuffle(positive)
    rng.shuffle(negative)
    n_pos_val = max(2, round(len(positive) * val_fraction))
    n_neg_val = max(2, round(len(negative) * val_fraction))
    val_groups = set(positive[:n_pos_val]) | set(negative[:n_neg_val])
    train = [row for row in examples if row["group"] not in val_groups]
    val = [row for row in examples if row["group"] in val_groups]
    return train, val


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=SFT_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="debug: cap total examples")
    args = parser.parse_args(argv)

    chunks = read_jsonl(CHUNKS_PATH)
    sft_chunks = [chunk for chunk in chunks if chunk["role"] == "sft"]
    if not sft_chunks:
        print("no SFT-role chunks; run `make chunk` first", file=sys.stderr)
        return 1
    retriever = Retriever.from_chunks(sft_chunks)
    examples = build_examples(chunks, retriever, k=config["generation"]["top_k_passages"])
    if args.limit:
        examples = examples[: args.limit]
    train, val = split_examples(examples, val_fraction=args.val_fraction)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    manifest = {
        "seed": SEED,
        "source": "data/chunks.jsonl",
        "sft_chunks": [chunk["chunk_id"] for chunk in sft_chunks],
        "held_out_pages_used": False,
        "n_train": len(train),
        "n_val": len(val),
        "n_answerable": sum(row["category"] == "answerable" for row in examples),
        "n_abstain": sum(row["category"] == "abstain" for row in examples),
        "max_context_chars": MAX_CONTEXT_CHARS,
        "k": config["generation"]["top_k_passages"],
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {len(train)} train / {len(val)} val examples "
        f"({manifest['n_answerable']} answerable, {manifest['n_abstain']} abstain)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
