"""Retrieval-only evaluation: Recall@k and MRR against hand-checked probes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from src.common import (
    CHUNKS_PATH,
    INDEX_DIR,
    PROBES_PATH,
    RESULTS_DIR,
    ROOT,
    docset,
    read_jsonl,
)
from src.retrieve import Retriever


def is_relevant(chunk: dict, gold: list[dict]) -> bool:
    for target in gold:
        if chunk["page_id"] != target["page_id"]:
            continue
        anchor = target.get("anchor")
        if anchor is None or anchor in chunk["section_anchors"]:
            return True
    return False


def on_page(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["page_id"] == target["page_id"] for target in gold)


def validate_probes(probes: list[dict], chunks: list[dict]) -> list[str]:
    problems: list[str] = []
    known = {(chunk["page_id"], chunk["anchor"]) for chunk in chunks}
    pages = {chunk["page_id"] for chunk in chunks}
    seen: set[str] = set()
    for probe in probes:
        qid = probe.get("qid", "")
        if not qid or qid in seen:
            problems.append(f"missing or duplicate qid: {qid!r}")
        seen.add(qid)
        if not probe.get("question", "").strip():
            problems.append(f"{qid}: empty question")
        if not probe.get("gold"):
            problems.append(f"{qid}: no gold targets")
        for target in probe.get("gold", []):
            page, anchor = target.get("page_id"), target.get("anchor")
            if page not in pages:
                problems.append(f"{qid}: unknown page {page!r}")
            elif anchor is not None and not any(
                c["page_id"] == page and anchor in c["section_anchors"] for c in chunks
            ):
                problems.append(f"{qid}: unknown section {page}#{anchor}")
    return problems


def evaluate(retriever: Retriever, probes: list[dict], ks: tuple[int, ...]) -> list[dict]:
    rows = []
    max_k = max(ks)
    for probe in probes:
        results = retriever.search(probe["question"], k=max_k)
        relevant_ranks = [
            rank for rank, chunk in enumerate(results, start=1) if is_relevant(chunk, probe["gold"])
        ]
        page_ranks = [
            rank for rank, chunk in enumerate(results, start=1) if on_page(chunk, probe["gold"])
        ]
        rank = relevant_ranks[0] if relevant_ranks else None
        page_rank = page_ranks[0] if page_ranks else None
        top = results[0] if results else None
        rows.append(
            {
                "qid": probe["qid"],
                "topic": probe.get("topic", "unknown"),
                "question": probe["question"],
                "rank": rank,
                "page_rank": page_rank,
                "hit_at": {str(k): bool(rank and rank <= k) for k in ks},
                "page_hit_at": {str(k): bool(page_rank and page_rank <= k) for k in ks},
                "top1": None
                if top is None
                else {"page_id": top["page_id"], "anchor": top["anchor"], "score": top["score"]},
                "candidates": [
                    f"{c['page_id']}#{c['anchor']} ({c['score']})" for c in results[:3]
                ],
                "note": probe.get("note"),
            }
        )
    return rows


def aggregate(rows: list[dict], ks: tuple[int, ...], key: str = "hit_at") -> dict:
    n = len(rows)
    out = {"n": n}
    for k in ks:
        out[f"recall@{k}"] = round(sum(row[key][str(k)] for row in rows) / n, 3) if n else 0.0
    ranks = [row["rank"] for row in rows if row["rank"]]
    out["mrr@5"] = round(sum(1 / rank for rank in ranks if rank <= max(ks)) / n, 3) if n else 0.0
    out["median_rank"] = statistics.median(ranks) if ranks else None
    return out


def markdown(rows: list[dict], ks: tuple[int, ...], info: dict, meta: dict) -> str:
    overall = aggregate(rows, ks)
    overall_pages = aggregate(rows, ks, key="page_hit_at")
    topics = sorted({row["topic"] for row in rows})
    topic_rows = {topic: [r for r in rows if r["topic"] == topic] for topic in topics}

    lines = [
        "# Retrieval baseline: BM25",
        "",
        f"- Docs: `{info['repo']}` @ `{info['tag']}` (`{info['commit_sha'][:12]}`)",
        f"- Corpus: {meta['chunk_count']} chunks across 7 pages; seeds fixed; no LLM involved",
        f"- Probes: {len(rows)} hand-written retrieval probes (not the evaluation set)",
        f"- Index: BM25 k1={meta['k1']}, b={meta['b']}, avgdl={meta['avgdl']}",
        "",
        "Recall counts a hit when the retrieved chunk belongs to any gold page and the gold",
        "section is on that chunk's heading path. Page recall is a looser diagnostic.",
        "",
        "## Overall",
        "",
        "| metric | " + " | ".join(f"@{k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
        "| section recall | " + " | ".join(str(overall[f"recall@{k}"]) for k in ks) + " |",
        "| page recall | " + " | ".join(str(overall_pages[f"recall@{k}"]) for k in ks) + " |",
        "",
        f"MRR@{max(ks)} = **{overall['mrr@5']}**, median rank = {overall['median_rank']}, n = {overall['n']}",
        "",
        "## Per topic",
        "",
        "| topic | n | " + " | ".join(f"recall@{k}" for k in ks) + " | MRR@5 |",
        "|---|---|" + "---|" * len(ks) + "---|",
    ]
    for topic, topic_items in topic_rows.items():
        agg = aggregate(topic_items, ks)
        lines.append(
            f"| {topic} | {agg['n']} | "
            + " | ".join(str(agg[f"recall@{k}"]) for k in ks)
            + f" | {agg['mrr@5']} |"
        )

    lines += [
        "",
        "## Per probe",
        "",
        "| qid | topic | question | gold rank | top-1 result |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        top = row["top1"]
        top_text = "none" if not top else f"{top['page_id']}#{top['anchor']}"
        rank = row["rank"] if row["rank"] is not None else "miss"
        question = row["question"].replace("|", "\\|")
        lines.append(f"| {row['qid']} | {row['topic']} | {question} | {rank} | {top_text} |")

    misses = [row for row in rows if row["rank"] is None]
    lines += ["", "## Misses", ""]
    if not misses:
        lines.append("None.")
    for row in misses:
        lines.append(f"- **{row['qid']}** ({row['topic']}): {row['question']}")
        lines.append(f"  - candidates: {'; '.join(row['candidates']) or 'none'}")
    lines += [
        "",
        "## Known weaknesses exposed here",
        "",
        "- No stemming: `tune` does not match `tuning`, `sample` does not match `sampling`.",
        "- Synonyms are invisible: `varied text` never reaches `sampling`.",
        "- Section-level gold can be beaten by a related section on the same page; page recall",
        "  is reported separately so that distinction stays visible.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, default=PROBES_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--out-prefix", type=Path, default=RESULTS_DIR / "retrieval_baseline")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args(argv)

    chunks = read_jsonl(CHUNKS_PATH)
    probes = read_jsonl(args.probes)
    if not chunks:
        print("no chunks; run `make chunk` first", file=sys.stderr)
        return 1
    if not probes:
        print(f"no probes found at {args.probes}", file=sys.stderr)
        return 1
    problems = validate_probes(probes, chunks)
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    retriever = Retriever.load(args.index_dir)
    ks = tuple(sorted(set(args.k)))
    rows = evaluate(retriever, probes, ks)
    overall = aggregate(rows, ks)
    per_topic = {
        topic: aggregate([r for r in rows if r["topic"] == topic], ks)
        for topic in sorted({r["topic"] for r in rows})
    }
    info = docset()
    meta = retriever.meta
    payload = {
        "method": "bm25",
        "docset": {"tag": info["tag"], "commit_sha": info["commit_sha"]},
        "index": {key: meta[key] for key in ("chunk_count", "k1", "b", "avgdl", "vocab_size")},
        "probes": str(args.probes.relative_to(ROOT)),
        "ks": list(ks),
        "overall": overall,
        "per_topic": per_topic,
        "rows": rows,
    }
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = markdown(rows, ks, info, meta)
    args.out_prefix.with_suffix(".md").write_text(report, encoding="utf-8")

    print(
        f"section recall: "
        + " ".join(f"@{k}={overall[f'recall@{k}']}" for k in ks)
        + f"  MRR@{max(ks)}={overall['mrr@5']}  n={overall['n']}"
    )
    for topic, agg in per_topic.items():
        print(
            f"  {topic:12} n={agg['n']:2}  "
            + " ".join(f"@{k}={agg[f'recall@{k}']}" for k in ks)
            + f"  mrr={agg['mrr@5']}"
        )
    print(f"wrote {args.out_prefix.with_suffix('.md').relative_to(ROOT)} and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
