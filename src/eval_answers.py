"""Run grounded RAG answers over the evaluation split and score the contract.

This scores only what can be checked automatically: format, abstention behaviour,
citation validity and whether a cited passage is a gold passage. Support and factual
correctness are left to the human label sheet (``results/label_sheet_baseline.csv``).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from src.common import (
    QUESTIONS_PATH,
    RESULTS_DIR,
    ROOT,
    docset,
    load_config,
    read_jsonl,
    write_jsonl,
)
from src.eval_retrieval import is_relevant, on_page
from src.generate import (
    ABSTAIN_TEXT,
    load_hf,
    make_generate_fn,
    parse_response,
    resolve_model,
)
from src.label_sheet import COLUMNS, render_gold
from src.retrieve import Retriever

AUTO_COLUMNS = ["status", "auto_citation_hit", "format_ok", "latency_s", "repairs", "notes_auto"]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _aggregate(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["category"] == "answerable"]
    negative = [row for row in rows if row["expected"] == "abstain"]
    answered = [row for row in rows if row["status"] == "answered"]
    precisions = [row["citation_precision"] for row in answered if row["citation_precision"] is not None]
    latencies = [row["latency_s"] for row in rows]
    return {
        "n": len(rows),
        "behavior_accuracy": _mean([row["behavior_ok"] for row in rows]),
        "answered_rate": _mean([row["status"] == "answered" for row in rows]),
        "abstained_rate": _mean([row["abstained"] for row in rows]),
        "model_abstain_rate_on_negative": _mean([row["model_abstained"] for row in negative]),
        "delivered_abstention_rate_on_negative": _mean([row["abstained"] for row in negative]),
        "over_answer_rate_on_negative": _mean([row["over_answer"] for row in negative]),
        "parser_save_rate_on_negative": _mean([row["forced_abstain_attempt"] for row in negative]),
        "false_abstain_rate_on_answerable": _mean([row["abstained"] for row in answerable]),
        "grounded_answer_rate": _mean(
            [row["status"] == "answered" and bool(row["citation_hit"]) for row in answerable]
        ),
        "citation_hit_rate": _mean([bool(row["citation_hit"]) for row in answerable]),
        "page_citation_hit_rate": _mean([bool(row["page_citation_hit"]) for row in answerable]),
        "mean_citation_precision": _mean(precisions),
        "format_rate": _mean([row["format_ok"] for row in rows]),
        "retrieval_hit_rate": _mean([bool(row["retrieval_hit"]) for row in answerable]),
        "repairs": sum(len(row["repairs"]) for row in rows),
        "forced_abstain": sum(row["status"] == "forced_abstain" for row in rows),
        "mean_latency_s": _mean(latencies),
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
    }


def _source_rows(parsed: dict, retriever: Retriever, gold: list[dict]) -> list[dict]:
    rows = []
    for source in parsed["sources"]:
        chunk = retriever.by_id[source["chunk_id"]]
        rows.append(
            {
                **source,
                "relevant": is_relevant(chunk, gold),
                "on_page": on_page(chunk, gold),
            }
        )
    return rows


def _score_row(question: dict, retriever: Retriever, generate_fn, k: int) -> dict:
    passages = retriever.search(question["question"], k=k)
    gold = question["gold"]
    retrieval_hit = any(is_relevant(chunk, gold) for chunk in passages)
    retrieval_page_hit = any(on_page(chunk, gold) for chunk in passages)

    latency = 0.0
    error = None
    if not passages:
        raw = ""
        parsed = parse_response(raw, [])
        parsed["reason"] = "no_retrieved_passages"
    else:
        started = time.perf_counter()
        try:
            raw = generate_fn(question["question"], passages)
        except Exception as exc:  # keep a 43-question run alive across one bad call
            raw = ""
            error = f"{type(exc).__name__}: {exc}"
        latency = time.perf_counter() - started
        parsed = parse_response(raw, passages)
        if error:
            parsed["status"] = "forced_abstain"
            parsed["reason"] = "generation_error"

    sources = _source_rows(parsed, retriever, gold) if parsed["sources"] else []
    cited_relevant = [source for source in sources if source["relevant"]]
    expected_abstain = question["expected"] == "abstain"
    delivered_answered = parsed["status"] == "answered"
    delivered_abstention = not delivered_answered
    model_abstained = parsed["status"] == "abstained"
    forced_attempt = parsed["status"] == "forced_abstain" and parsed.get("reason") not in {
        "no_retrieved_passages",
        "generation_error",
    }
    behavior_ok = delivered_answered if not expected_abstain else model_abstained
    over_answer = expected_abstain and delivered_answered
    false_abstain = not expected_abstain and delivered_abstention
    top = passages[0] if passages else None

    return {
        "qid": question["qid"],
        "split": question["split"],
        "topic": question["topic"],
        "category": question["category"],
        "expected": question["expected"],
        "question": question["question"],
        "reference": question["reference"],
        "gold": render_gold(gold),
        "status": parsed["status"],
        "reason": parsed.get("reason"),
        "error": error,
        "abstained": delivered_abstention,
        "model_abstained": model_abstained,
        "forced_abstain_attempt": forced_attempt,
        "expected_abstain": expected_abstain,
        "behavior_ok": behavior_ok,
        "over_answer": over_answer,
        "false_abstain": false_abstain,
        "format_ok": parsed["format_ok"],
        "citations": parsed["citations"],
        "sources": sources,
        "invalid_source_entries": parsed["invalid_source_entries"],
        "repairs": parsed["repairs"],
        "unverified_urls": parsed["unverified_urls"],
        "citation_hit": any(source["relevant"] for source in sources) if question["category"] == "answerable" else None,
        "page_citation_hit": any(source["on_page"] for source in sources) if question["category"] == "answerable" else None,
        "citation_precision": round(len(cited_relevant) / len(sources), 3) if sources else None,
        "retrieval_hit": retrieval_hit,
        "retrieval_page_hit": retrieval_page_hit,
        "retrieved_top1": None if top is None else f"{top['page_id']}#{top['anchor']}",
        "retrieved": [
            {"chunk_id": chunk["chunk_id"], "page_id": chunk["page_id"], "anchor": chunk["anchor"], "score": chunk["score"]}
            for chunk in passages
        ],
        "latency_s": round(latency, 2),
        "answer": parsed["answer"],
        "extracted_answer": parsed.get("extracted_answer", parsed["answer"]),
        "raw": raw,
    }


def _table(overall: dict) -> list[str]:
    keys = [
        ("n", "n"),
        ("behavior_accuracy", "behaviour accuracy"),
        ("answered_rate", "answered"),
        ("grounded_answer_rate", "grounded answer (answerable)"),
        ("citation_hit_rate", "citation hits a gold passage (answerable)"),
        ("page_citation_hit_rate", "citation hits a gold page (answerable)"),
        ("mean_citation_precision", "mean citation precision (answered)"),
        ("model_abstain_rate_on_negative", "model abstains on negative questions"),
        ("delivered_abstention_rate_on_negative", "delivered abstention on negative questions"),
        ("over_answer_rate_on_negative", "delivered answer on negative questions"),
        ("parser_save_rate_on_negative", "parser saves on negative questions"),
        ("false_abstain_rate_on_answerable", "false abstain (answerable)"),
        ("format_rate", "exact format"),
        ("retrieval_hit_rate", "section retrieval hit@k (answerable)"),
        ("repairs", "citation repairs"),
        ("forced_abstain", "forced abstentions"),
        ("median_latency_s", "median latency (s)"),
    ]
    lines = ["| metric | value |", "|---|---|"]
    for key, label in keys:
        value = overall.get(key)
        lines.append(f"| {label} | {value} |")
    return lines


def markdown(payload: dict) -> str:
    overall = payload["overall"]
    lines = [
        "# Grounded answer baseline (untuned)",
        "",
        f"- Docs: `{payload['docset']['repo']}` @ `{payload['docset']['tag']}` (`{payload['docset']['commit_sha'][:12]}`)",
        f"- Model: `{payload['model_id']}`"
        + (f" @ `{payload['model_commit'][:12]}`" if payload.get("model_commit") else "")
        + f" ({payload['quantization']}), greedy decoding, max_new_tokens={payload['max_new_tokens']}",
        f"- Split: **{payload['split']}**, n={payload['n']}, k={payload['k']} passages",
        "- Contract: 2-6 sentences, `[n]` citations after claims, a `Sources:` block with the exact passage URL,",
        f"  or exactly `{ABSTAIN_TEXT}`",
        "- Automated only: format, abstention, citation validity. Support/factual correctness: see the label sheet.",
        "",
        "## Overall",
        "",
        *_table(overall),
        "",
        "## By category",
        "",
        "| category | n | behaviour | answered | grounded | format | citation precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for category, agg in payload["per_category"].items():
        lines.append(
            f"| {category} | {agg['n']} | {agg['behavior_accuracy']} | {agg['answered_rate']} | "
            f"{agg['grounded_answer_rate']} | {agg['format_rate']} | {agg['mean_citation_precision']} |"
        )
    lines += [
        "",
        "## By topic",
        "",
        "| topic | n | behaviour | answered | citation hit | format |",
        "|---|---|---|---|---|---|",
    ]
    for topic, agg in payload["per_topic"].items():
        lines.append(
            f"| {topic} | {agg['n']} | {agg['behavior_accuracy']} | {agg['answered_rate']} | "
            f"{agg['citation_hit_rate']} | {agg['format_rate']} |"
        )

    rows = payload["rows"]
    over = [row for row in rows if row["over_answer"]]
    saved = [row for row in rows if row["expected_abstain"] and row["forced_abstain_attempt"]]
    false = [row for row in rows if row["false_abstain"]]
    ungrounded = [row for row in rows if row["status"] == "answered" and row["citation_hit"] is False]
    format_fail = [row for row in rows if not row["format_ok"]]

    lines += ["", "## Over-answered negative questions", ""]
    lines += [f"None."] if not over else [
        f"- **{row['qid']}** ({row['category']}): {row['question']}\n  - answer: {row['answer'][:240]}"
        for row in over
    ]
    lines += ["", "## Parser saves on negative questions (model tried to answer)", ""]
    lines += [f"None."] if not saved else [
        f"- **{row['qid']}** ({row['category']}): {row['reason']}; citations={row['citations']}; "
        f"extracted: {row['extracted_answer'][:200]}"
        for row in saved
    ]
    lines += ["", "## False abstentions on answerable questions", ""]
    lines += [f"None."] if not false else [
        f"- **{row['qid']}** ({row['topic']}): {row['question']}\n  - reason: {row['reason']}, retrieved: {row['retrieved_top1']}"
        for row in false
    ]
    lines += ["", "## Answered but no cited gold passage", ""]
    lines += [f"None."] if not ungrounded else [
        f"- **{row['qid']}** ({row['topic']}): cited {row['citations']}, gold {row['gold']}"
        for row in ungrounded
    ]
    lines += ["", "## Format issues", ""]
    lines += [f"None."] if not format_fail else [
        f"- **{row['qid']}**: {row['reason'] or 'format'}; repairs={row['repairs']}; invalid={row['invalid_source_entries']}"
        for row in format_fail
    ]
    lines += [
        "",
        "## Method notes",
        "",
        "- Retrieval is the untouched BM25 baseline; this run isolates generation behaviour.",
        "- A forced abstention means the parser refused the output (no valid citation, an invented",
        "  URL, or a generation error), not that the model chose to abstain.",
        "- Invalid URLs are replaced with the canonical passage URL and counted as repairs; the",
        "  output can never contain a URL that is absent from the retrieved passages.",
        "- These numbers are a floor for the untuned model, not a RAG quality claim.",
        "",
    ]
    return "\n".join(lines)


def _bit(value) -> str:
    if value is None:
        return ""
    return "1" if value else "0"


def write_label_sheet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = COLUMNS + AUTO_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "qid": row["qid"],
                    "split": row["split"],
                    "topic": row["topic"],
                    "category": row["category"],
                    "expected": row["expected"],
                    "question": row["question"],
                    "gold": row["gold"],
                    "reference": row["reference"],
                    "retrieved_top1": row["retrieved_top1"] or "",
                    "retrieval_hit": _bit(row["retrieval_hit"]),
                    "abstained": _bit(row["abstained"]),
                    "status": row["status"],
                    "auto_citation_hit": _bit(row["citation_hit"]),
                    "format_ok": _bit(row["format_ok"]),
                    "latency_s": row["latency_s"],
                    "repairs": len(row["repairs"]),
                    "notes_auto": row["reason"] or row["error"] or "",
                }
            )


def run(
    retriever: Retriever,
    questions: list[dict],
    generate_fn,
    *,
    k: int,
    split: str,
    limit: int,
    out_prefix: Path,
    label_out: Path,
    model_id: str,
    quantization: str,
    max_new_tokens: int,
    model_commit: str | None = None,
    log=sys.stderr,
) -> dict:
    selected = [
        question
        for question in sorted(questions, key=lambda item: item["qid"])
        if split == "all" or question["split"] == split
    ]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise SystemExit(f"no questions for split={split!r}")

    rows = []
    for index, question in enumerate(selected, start=1):
        row = _score_row(question, retriever, generate_fn, k)
        rows.append(row)
        print(
            f"[{index}/{len(selected)}] {row['qid']} {row['status']} "
            f"behaviour={'ok' if row['behavior_ok'] else 'FAIL'} {row['latency_s']}s",
            file=log,
        )

    payload = {
        "method": "grounded_rag_untuned",
        "model_id": model_id,
        "model_commit": model_commit,
        "quantization": quantization,
        "max_new_tokens": max_new_tokens,
        "docset": {
            "repo": load_config()["docs"]["repo"],
            "tag": docset()["tag"],
            "commit_sha": docset()["commit_sha"],
        },
        "split": split,
        "k": k,
        "n": len(rows),
        "overall": _aggregate(rows),
        "per_category": {
            category: _aggregate([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        },
        "per_topic": {
            topic: _aggregate([row for row in rows if row["topic"] == topic])
            for topic in sorted({row["topic"] for row in rows})
        },
        "rows": rows,
    }

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out_prefix.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    margin = out_prefix.stem.removeprefix("answers_")
    write_jsonl(out_prefix.parent / f"predictions_{margin}.jsonl", rows)
    write_label_sheet(rows, label_out)
    return payload


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["test", "dev", "all"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("-k", type=int, default=config["generation"]["top_k_passages"])
    parser.add_argument("--model", default="base", help="base | smoke | fallback | HF model id")
    parser.add_argument("--quantize", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=config["generation"]["max_new_tokens"])
    parser.add_argument("--out-prefix", type=Path, default=RESULTS_DIR / "answers_baseline")
    parser.add_argument("--labels", type=Path, default=RESULTS_DIR / "label_sheet_baseline.csv")
    args = parser.parse_args(argv)

    questions = read_jsonl(QUESTIONS_PATH)
    if not questions:
        print(f"no questions at {QUESTIONS_PATH}", file=sys.stderr)
        return 1
    try:
        retriever = Retriever.load()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    model_id = resolve_model(args.model, config)
    if args.quantize == "auto":
        import torch

        quantize = torch.cuda.is_available()
    else:
        quantize = args.quantize == "on"
    print(
        f"model: {model_id} | quantize: {quantize} | split: {args.split} | k: {args.k}",
        file=sys.stderr,
    )
    model, tokenizer = load_hf(model_id, quantize_4bit=quantize)
    model_commit = getattr(model.config, "_commit_hash", None) or getattr(
        tokenizer, "_commit_hash", None
    )
    generate_fn = make_generate_fn(model, tokenizer, max_new_tokens=args.max_new_tokens)
    payload = run(
        retriever,
        questions,
        generate_fn,
        k=args.k,
        split=args.split,
        limit=args.limit,
        out_prefix=args.out_prefix,
        label_out=args.labels,
        model_id=model_id,
        quantization="4-bit NF4" if quantize else "full precision",
        max_new_tokens=args.max_new_tokens,
        model_commit=model_commit,
    )
    overall = payload["overall"]
    print(
        f"behaviour={overall['behavior_accuracy']} answered={overall['answered_rate']} "
        f"grounded={overall['grounded_answer_rate']} format={overall['format_rate']} "
        f"n={overall['n']}"
    )
    print(
        f"wrote {_display(args.out_prefix.with_suffix('.md'))} and label sheet {_display(args.labels)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
