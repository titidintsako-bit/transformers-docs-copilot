"""Split pinned pages into breadcrumbed section chunks for retrieval."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

from src.common import CHUNKS_PATH, ROOT, docset, load_config, read_manifest, write_jsonl
from src.text import estimate_tokens, iter_paragraphs, split_sections


def pack_paragraphs(paragraphs: list[str], budget: int, overlap: int) -> list[list[str]]:
    """Greedy paragraph packing with paragraph-aligned overlap.

    Guarantees forward progress: after the first window, at least one paragraph is
    always new, so a single oversized paragraph cannot stall the loop.
    """
    if not paragraphs:
        return []
    if len(paragraphs) == 1 or sum(estimate_tokens(p) for p in paragraphs) <= budget:
        return [list(paragraphs)]

    windows: list[list[str]] = []
    index = 0
    total = len(paragraphs)
    while index < total:
        start = index
        current: list[str] = []
        used = 0
        while index < total:
            cost = estimate_tokens(paragraphs[index])
            if current and used + cost > budget:
                break
            current.append(paragraphs[index])
            used += cost
            index += 1
        windows.append(current)
        if index >= total:
            break
        advanced = index - start
        back: list[str] = []
        back_tokens = 0
        for paragraph in reversed(current):
            cost = estimate_tokens(paragraph)
            if back_tokens + cost > overlap:
                break
            back.insert(0, paragraph)
            back_tokens += cost
        rewind = min(len(back), max(0, advanced - 1))
        index -= rewind
    return windows


def build_chunks(manifest: list[dict], *, tag: str, max_tokens: int, overlap: float) -> list[dict]:
    chunks: list[dict] = []
    for page in manifest:
        raw = (ROOT / page["raw_path"]).read_text(encoding="utf-8")
        title, sections = split_sections(raw)
        if title != page["title"]:
            raise RuntimeError(f"{page['page_id']}: title drifted ({title!r})")
        order = 0
        for section in sections:
            body = section["body"].strip()
            if not body:
                continue
            path = section["path"] or [page["title"]]
            breadcrumb = f"Transformers {tag} > {page['title']}"
            if path != [page["title"]]:
                breadcrumb += " > " + " > ".join(path)
            budget = max(120, max_tokens - estimate_tokens(breadcrumb))
            paragraphs = [text for text, _ in iter_paragraphs(body)]
            if not paragraphs:
                continue
            for part, window in enumerate(
                pack_paragraphs(paragraphs, budget, max(0, round(budget * overlap)))
            ):
                body_text = "\n\n".join(window).strip()
                if not body_text:
                    continue
                text = f"{breadcrumb}\n\n{body_text}"
                anchor = section["anchor"]
                slug = anchor or "_intro"
                chunks.append(
                    {
                        "chunk_id": f"{page['page_id']}:{slug}:{part}",
                        "page_id": page["page_id"],
                        "page_title": page["title"],
                        "role": page["role"],
                        "topics": page["topics"],
                        "anchor": anchor,
                        "heading": section["heading"],
                        "section": " > ".join(path),
                        "section_path": path,
                        "section_anchors": section["anchors"],
                        "order": order,
                        "part": part,
                        "breadcrumb": breadcrumb,
                        "url": page["url"] + (f"#{anchor}" if anchor else ""),
                        "docs_url": page["docs_url"],
                        "n_tokens_est": estimate_tokens(text),
                        "chars": len(text),
                        "text": text,
                    }
                )
                order += 1
    return chunks


def validate_chunks(chunks: list[dict], manifest: list[dict], max_tokens: int) -> list[str]:
    problems: list[str] = []
    ids = Counter(chunk["chunk_id"] for chunk in chunks)
    problems += [f"duplicate chunk_id: {cid}" for cid, count in ids.items() if count > 1]

    anchors = {page["page_id"]: {h["anchor"] for h in page["headings"]} for page in manifest}
    for chunk in chunks:
        if chunk["anchor"] is not None and chunk["anchor"] not in anchors[chunk["page_id"]]:
            problems.append(f"{chunk['chunk_id']}: unknown anchor {chunk['anchor']!r}")
        if not chunk["text"].startswith("Transformers "):
            problems.append(f"{chunk['chunk_id']}: missing breadcrumb")
        if not chunk["url"].startswith("https://github.com/"):
            problems.append(f"{chunk['chunk_id']}: bad url")
        if not chunk["text"].strip():
            problems.append(f"{chunk['chunk_id']}: empty text")

    for page in manifest:
        page_chunks = [c for c in chunks if c["page_id"] == page["page_id"]]
        if not page_chunks:
            problems.append(f"{page['page_id']}: no chunks produced")
        raw = (ROOT / page["raw_path"]).read_text(encoding="utf-8")
        _, sections = split_sections(raw)
        with_body = {s["anchor"] for s in sections if s["body"].strip()}
        produced = {c["anchor"] for c in page_chunks}
        for anchor in with_body - produced:
            problems.append(f"{page['page_id']}: section {anchor!r} produced no chunk")

    oversize = [c for c in chunks if c["n_tokens_est"] > max_tokens * 1.4]
    for chunk in oversize:
        problems.append(
            f"{chunk['chunk_id']}: oversize chunk ({chunk['n_tokens_est']} est tokens)"
        )
    return problems


def summarize(chunks: list[dict], manifest: list[dict], max_tokens: int, overlap: float) -> None:
    sizes = sorted(c["n_tokens_est"] for c in chunks)
    print(f"chunks: {len(chunks)}  (max_tokens={max_tokens}, overlap={overlap:.0%})")
    print(
        f"tokens est: min={sizes[0]} p50={statistics.median(sizes):.0f} "
        f"p90={sizes[int(len(sizes) * 0.9) - 1]} max={sizes[-1]}"
    )
    print(f"{'page':26} {'role':5} {'chunks':>6} {'sections':>8}")
    for page in manifest:
        page_chunks = [c for c in chunks if c["page_id"] == page["page_id"]]
        sections = len({c["anchor"] for c in page_chunks})
        print(f"{page['page_id']:26} {page['role']:5} {len(page_chunks):>6} {sections:>8}")
    multi = sum(1 for count in Counter((c["page_id"], c["anchor"]) for c in chunks).values() if count > 1)
    print(f"sections split across multiple chunks: {multi}")
    print(f"sft chunks: {sum(c['role'] == 'sft' for c in chunks)} | "
          f"test chunks: {sum(c['role'] == 'test' for c in chunks)}")


def main(argv: list[str] | None = None) -> int:
    config = load_config()
    retrieval = config["retrieval"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=retrieval["chunk_tokens"])
    parser.add_argument("--overlap", type=float, default=retrieval["chunk_overlap"])
    parser.add_argument("--out", type=Path, default=CHUNKS_PATH)
    args = parser.parse_args(argv)

    manifest = read_manifest()
    if not manifest:
        print("run `make fetch` first: manifest is empty", file=sys.stderr)
        return 1
    info = docset()
    chunks = build_chunks(
        manifest,
        tag=info["tag"],
        max_tokens=args.max_tokens,
        overlap=args.overlap,
    )
    problems = validate_chunks(chunks, manifest, args.max_tokens)
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1
    write_jsonl(args.out, chunks)
    summarize(chunks, manifest, args.max_tokens, args.overlap)
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
