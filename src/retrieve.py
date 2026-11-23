"""Load the index and retrieve chunks for a query."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.bm25 import BM25
from src.common import CHUNKS_PATH, INDEX_DIR, ROOT, read_jsonl, sha256_file
from src.index import index_files


class Retriever:
    def __init__(self, chunks: list[dict], model: BM25, meta: dict | None = None):
        self.chunks = chunks
        self.by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.model = model
        self.meta = meta or {}

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "Retriever":
        model = BM25([chunk["text"] for chunk in chunks], [chunk["chunk_id"] for chunk in chunks])
        return cls(chunks, model)

    @classmethod
    def load(cls, index_dir: Path = INDEX_DIR) -> "Retriever":
        model_path, meta_path = index_files(index_dir)
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError("index missing; run `make index` first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if sha256_file(CHUNKS_PATH) != meta["chunk_sha256"]:
            raise RuntimeError("index is stale (chunks changed); run `make index`")
        chunks = read_jsonl(CHUNKS_PATH)
        model = BM25.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
        return cls(chunks, model, meta)

    def search(self, query: str, k: int = 5, page_id: str | None = None) -> list[dict]:
        wanted = k if page_id is None else k * 6
        results = []
        for chunk_id, score in self.model.search(query, k=wanted):
            chunk = self.by_id[chunk_id]
            if page_id is not None and chunk["page_id"] != page_id:
                continue
            results.append({**chunk, "score": round(score, 4)})
            if len(results) >= k:
                break
        return results


def render(result: dict) -> str:
    body = result["text"].split("\n\n", 1)[1].replace("\n", " ")
    snippet = body[:220] + ("..." if len(body) > 220 else "")
    return (
        f"[{result['score']:.3f}] {result['page_id']} | {result['section']}\n"
        f"    {result['url']}\n"
        f"    {snippet}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="search query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--page", help="restrict to one page id")
    parser.add_argument("--json", action="store_true", help="emit full chunk JSON")
    args = parser.parse_args(argv)

    try:
        retriever = Retriever.load()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    results = retriever.search(args.query, k=args.k, page_id=args.page)
    if not results:
        print("no results (a zero-score miss is reported as a miss, not a best guess)")
        return 0
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {render(result)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
