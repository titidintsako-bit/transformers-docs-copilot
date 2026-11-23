"""Generate the blank human-scoring sheet for the evaluation set."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from src.common import LABEL_SHEET_PATH, QUESTIONS_PATH, read_jsonl

COLUMNS = [
    "qid",
    "split",
    "topic",
    "category",
    "expected",
    "question",
    "gold",
    "reference",
    "retrieved_top1",
    "retrieval_hit",
    "abstained",
    "supported_0_1_2",
    "citation_precision_0_1",
    "citation_recall_0_1",
    "notes",
]


def render_gold(gold: list[dict]) -> str:
    if not gold:
        return ""
    return "; ".join(
        f"{target['page_id']}#{target.get('anchor') or ''}" for target in gold
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--out", type=Path, default=LABEL_SHEET_PATH)
    args = parser.parse_args(argv)

    questions = read_jsonl(args.questions)
    if not questions:
        print(f"no questions at {args.questions}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for question in sorted(questions, key=lambda item: (item["split"], item["qid"])):
            writer.writerow(
                {
                    "qid": question["qid"],
                    "split": question["split"],
                    "topic": question["topic"],
                    "category": question["category"],
                    "expected": question["expected"],
                    "question": question["question"],
                    "gold": render_gold(question["gold"]),
                    "reference": question["reference"],
                }
            )
    print(f"wrote {args.out} ({len(questions)} rows, {len(COLUMNS)} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
