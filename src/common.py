"""Shared paths and small IO helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.yaml"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.jsonl"
CHUNKS_PATH = ROOT / "data" / "chunks.jsonl"
INDEX_DIR = ROOT / "artifacts" / "index"
PROBES_PATH = ROOT / "data" / "eval" / "retrieval_probes.jsonl"
QUESTIONS_PATH = ROOT / "data" / "eval" / "questions.jsonl"
SPLIT_PATH = ROOT / "data" / "eval" / "split.json"
LABEL_SHEET_PATH = ROOT / "data" / "eval" / "label_sheet.csv"
RESULTS_DIR = ROOT / "results"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest() -> list[dict]:
    return read_jsonl(MANIFEST_PATH)


def docset() -> dict:
    path = ROOT / "data" / "raw" / "docset.json"
    if not path.exists():
        raise FileNotFoundError("run `make fetch` first: data/raw/docset.json is missing")
    return json.loads(path.read_text(encoding="utf-8"))
