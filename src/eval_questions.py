"""Validate the 55-question evaluation set and write the split manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.common import (
    CHUNKS_PATH,
    QUESTIONS_PATH,
    SPLIT_PATH,
    load_config,
    read_jsonl,
)

CATEGORIES = {"answerable", "unanswerable", "parametric"}
SPLITS = {"dev", "test"}


def validate(questions: list[dict], chunks: list[dict], quotas: dict) -> list[str]:
    problems: list[str] = []
    sft_pages = {chunk["page_id"] for chunk in chunks if chunk["role"] == "sft"}
    test_pages = {chunk["page_id"] for chunk in chunks if chunk["role"] == "test"}
    sections = {(chunk["page_id"], anchor) for chunk in chunks for anchor in chunk["section_anchors"]}

    qids = Counter(question.get("qid", "") for question in questions)
    problems += [f"duplicate qid: {qid}" for qid, count in qids.items() if count > 1]
    questions_text = Counter(question.get("question", "").strip() for question in questions)
    problems += [f"duplicate question text: {text[:60]!r}" for text, count in questions_text.items() if count > 1]

    for question in questions:
        qid = question.get("qid", "?")
        split = question.get("split")
        category = question.get("category")
        expected = question.get("expected")
        gold = question.get("gold", [])
        answerable = question.get("answerable")

        if split not in SPLITS:
            problems.append(f"{qid}: bad split {split!r}")
        if category not in CATEGORIES:
            problems.append(f"{qid}: bad category {category!r}")
        if expected not in {"answer", "abstain"}:
            problems.append(f"{qid}: bad expected {expected!r}")
        if not question.get("question", "").strip():
            problems.append(f"{qid}: empty question")
        if not question.get("reference", "").strip():
            problems.append(f"{qid}: missing reference (for scoring only)")
        if not question.get("topic", "").strip():
            problems.append(f"{qid}: missing topic")

        is_answerable = category == "answerable"
        if answerable is not is_answerable:
            problems.append(f"{qid}: answerable flag disagrees with category")
        if expected != ("answer" if is_answerable else "abstain"):
            problems.append(f"{qid}: expected behaviour disagrees with category")
        if is_answerable and not gold:
            problems.append(f"{qid}: answerable question has no gold targets")
        if not is_answerable and gold:
            problems.append(f"{qid}: abstain question must not have gold targets")
        if not is_answerable and question.get("topic") != "out-of-scope":
            problems.append(f"{qid}: abstain question should use topic out-of-scope")
        if not is_answerable and split != "test":
            problems.append(f"{qid}: abstain questions must be in the test split")

        allowed = sft_pages if split == "dev" else test_pages
        for target in gold:
            page, anchor = target.get("page_id"), target.get("anchor")
            if page not in allowed:
                problems.append(
                    f"{qid}: gold page {page!r} is not allowed in the {split} split"
                )
            if anchor is not None and (page, anchor) not in sections:
                problems.append(f"{qid}: unknown section {page}#{anchor}")

    counts = Counter(question.get("category") for question in questions)
    split_counts = Counter(question.get("split") for question in questions)
    expected_counts = {
        "total": quotas["total"],
        "dev": quotas["dev"],
        "test": quotas["test"],
        "answerable": quotas["answerable"],
        "unanswerable": quotas["unanswerable"],
        "parametric": quotas["parametric"],
    }
    actual_counts = {
        "total": len(questions),
        "dev": split_counts["dev"],
        "test": split_counts["test"],
        **{category: counts[category] for category in CATEGORIES},
    }
    for key, want in expected_counts.items():
        if actual_counts[key] != want:
            problems.append(f"count {key}: expected {want}, found {actual_counts[key]}")

    for page in sorted(test_pages):
        covered = sum(
            1
            for question in questions
            if question.get("split") == "test"
            and question.get("category") == "answerable"
            and any(target.get("page_id") == page for target in question.get("gold", []))
        )
        if covered < quotas["min_per_test_page"]:
            problems.append(
                f"test page {page}: only {covered} questions (need {quotas['min_per_test_page']})"
            )
    for page in sorted(sft_pages):
        covered = sum(
            1
            for question in questions
            if question.get("split") == "dev"
            and question.get("category") == "answerable"
            and any(target.get("page_id") == page for target in question.get("gold", []))
        )
        if covered < quotas["min_per_dev_page"]:
            problems.append(
                f"dev page {page}: only {covered} questions (need {quotas['min_per_dev_page']})"
            )

    return problems


def build_split(questions: list[dict], chunks: list[dict]) -> dict:
    sft_pages = sorted({chunk["page_id"] for chunk in chunks if chunk["role"] == "sft"})
    test_pages = sorted({chunk["page_id"] for chunk in chunks if chunk["role"] == "test"})
    by_topic = Counter(question["topic"] for question in questions)
    by_category = Counter(question["category"] for question in questions)
    return {
        "rule": (
            "Answerable dev questions use only sft pages; answerable test questions use only "
            "held-out pages. Unanswerable and parametric questions are test-only. Retrieval "
            "always indexes every page."
        ),
        "sft_pages": sft_pages,
        "test_pages": test_pages,
        "total": len(questions),
        "counts": {
            "dev": sum(question["split"] == "dev" for question in questions),
            "test": sum(question["split"] == "test" for question in questions),
            "answerable": by_category["answerable"],
            "unanswerable": by_category["unanswerable"],
            "parametric": by_category["parametric"],
        },
        "by_topic": dict(sorted(by_topic.items())),
        "dev_qids": [question["qid"] for question in questions if question["split"] == "dev"],
        "test_qids": [question["qid"] for question in questions if question["split"] == "test"],
        "test_answerable_qids": [
            question["qid"]
            for question in questions
            if question["split"] == "test" and question["category"] == "answerable"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--split-out", type=Path, default=SPLIT_PATH)
    parser.add_argument("--write", action="store_true", help="write the split manifest")
    args = parser.parse_args(argv)

    questions = read_jsonl(args.questions)
    chunks = read_jsonl(args.chunks)
    if not questions:
        print(f"no questions at {args.questions}", file=sys.stderr)
        return 1
    if not chunks:
        print("no chunks; run `make chunk` first", file=sys.stderr)
        return 1

    quotas = load_config()["eval"]
    problems = validate(questions, chunks, quotas)
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    split = build_split(questions, chunks)
    if args.write:
        args.split_out.parent.mkdir(parents=True, exist_ok=True)
        args.split_out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    print(
        f"OK: {split['total']} questions | dev {split['counts']['dev']} / "
        f"test {split['counts']['test']} | answerable {split['counts']['answerable']}, "
        f"unanswerable {split['counts']['unanswerable']}, parametric {split['counts']['parametric']}"
    )
    print("  by topic: " + ", ".join(f"{k}={v}" for k, v in split["by_topic"].items()))
    if args.write:
        print(f"wrote {args.split_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
