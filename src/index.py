"""Build the BM25 index from data/chunks.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.bm25 import BM25
from src.common import CHUNKS_PATH, INDEX_DIR, ROOT, docset, read_jsonl, sha256_file


def index_files(index_dir: Path) -> tuple[Path, Path]:
    return index_dir / "bm25.json", index_dir / "meta.json"


def build(index_dir: Path = INDEX_DIR) -> dict:
    chunks = read_jsonl(CHUNKS_PATH)
    if not chunks:
        raise RuntimeError("no chunks found; run `make chunk` first")
    model = BM25([chunk["text"] for chunk in chunks], [chunk["chunk_id"] for chunk in chunks])
    info = docset()
    index_dir.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = index_files(index_dir)
    model_path.write_text(json.dumps(model.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "chunk_count": len(chunks),
        "chunk_sha256": sha256_file(CHUNKS_PATH),
        "chunk_path": str(CHUNKS_PATH.relative_to(ROOT)),
        "k1": model.k1,
        "b": model.b,
        "avgdl": round(model.avgdl, 2),
        "vocab_size": len(model.idf),
        "docset_tag": info["tag"],
        "docset_commit": info["commit_sha"],
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def check(index_dir: Path = INDEX_DIR) -> int:
    _, meta_path = index_files(index_dir)
    if not meta_path.exists():
        print(f"FAIL: {meta_path} missing; run `make index`", file=sys.stderr)
        return 1
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not CHUNKS_PATH.exists():
        print("FAIL: chunks file missing", file=sys.stderr)
        return 1
    actual = sha256_file(CHUNKS_PATH)
    if actual != meta["chunk_sha256"]:
        print("FAIL: index is stale (chunks changed since indexing); run `make index`", file=sys.stderr)
        return 1
    if meta["chunk_count"] != len(read_jsonl(CHUNKS_PATH)):
        print("FAIL: chunk count mismatch", file=sys.stderr)
        return 1
    print(
        f"OK: {meta['chunk_count']} chunks, vocab {meta['vocab_size']}, "
        f"avgdl {meta['avgdl']}, {meta['docset_tag']} ({meta['docset_commit'][:12]})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    meta = build()
    print(
        f"indexed {meta['chunk_count']} chunks, vocab {meta['vocab_size']}, "
        f"avgdl {meta['avgdl']} -> {INDEX_DIR.relative_to(ROOT)}/bm25.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
