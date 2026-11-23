"""Fetch the pinned Transformers documentation subset and record provenance.

Usage:
    python -m src.fetch_docs            # download (idempotent) and write manifest
    python -m src.fetch_docs --check    # offline verification of local copies
    python -m src.fetch_docs --force    # re-download even if the pin matches
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.text import parse_markdown, sha256_text, strip_html_comments

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.yaml"
RAW_DIR = ROOT / "data" / "raw"
MD_DIR = RAW_DIR / "md"
MANIFEST_PATH = RAW_DIR / "manifest.jsonl"
DOCSET_PATH = RAW_DIR / "docset.json"
LICENSE_DOC = ROOT / "docs" / "DATA-LICENSE.md"
USER_AGENT = "transformers-docs-copilot/0.1 (provenance fetch)"
VALID_ROLES = {"sft", "test"}
REQUIRED_LICENSE_MARKERS = ("Apache License", "Version 2.0")


def http_get(url: str, *, retries: int = 3, timeout: int = 30) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise urllib.error.HTTPError(
                        url, response.status, response.reason, response.headers, None
                    )
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_error}")


def http_get_json(url: str) -> dict:
    return json.loads(http_get(url).decode("utf-8"))


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_commit(repo: str, tag: str) -> str:
    payload = http_get_json(f"https://api.github.com/repos/{repo}/commits/{tag}")
    sha = payload.get("sha", "")
    if len(sha) != 40:
        raise RuntimeError(f"could not resolve {repo}@{tag} to a commit (got {sha!r})")
    return sha


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_pages(pages: list[dict]) -> None:
    seen: set[str] = set()
    for page in pages:
        page_id = page.get("id", "")
        if not page_id or page_id in seen:
            raise RuntimeError(f"missing or duplicate page id: {page_id!r}")
        seen.add(page_id)
        if page.get("role") not in VALID_ROLES:
            raise RuntimeError(f"{page_id}: role must be one of {sorted(VALID_ROLES)}")
        if not page.get("path", "").endswith(".md"):
            raise RuntimeError(f"{page_id}: path must be a .md file")
        if not page.get("topics"):
            raise RuntimeError(f"{page_id}: at least one topic is required")
    if not any(p["role"] == "sft" for p in pages):
        raise RuntimeError("need at least one sft page")
    if not any(p["role"] == "test" for p in pages):
        raise RuntimeError("need at least one test page")


def license_document(docset: dict) -> str:
    rows = "\n".join(
        f"| `{page['page_id']}` | {page['title']} | {page['role']} | {page['bytes']:,} | "
        f"`{page['sha256'][:12]}` |"
        for page in docset["pages"]
    )
    return f"""# Data and reuse terms

## Source

- Repository: <https://github.com/{docset['repo']}>
- Pinned tag: `{docset['tag']}`
- Resolved commit: `{docset['commit_sha']}`
- Documentation root: `{docset['source_dir']}`
- Retrieved: {docset['retrieved_at']}

## License

The Transformers repository, including `{docset['source_dir']}`, is licensed under the
**Apache License, Version 2.0** ({docset['license_url']}). Every fetched page also carries an
Apache-2.0 copyright header, which is preserved verbatim in `data/raw/md/`. No per-page
exception was found for the files below; if one exists, exclude that page and regenerate.

### Attribution

Documentation text: © The Hugging Face Team, licensed under the Apache License, Version 2.0.
This project is not affiliated with or endorsed by Hugging Face.

### Changes

The source files are stored byte-for-byte. Downstream, this project removes license
comments before indexing, splits the text into sections, and adds breadcrumbs and IDs for
retrieval. Those transformations are documented in `data/raw/docset.json` and the chunking code.

## Fetched pages

| Page ID | Title | Role | Bytes | SHA-256 |
|---|---|---|---|---|
{rows}

`role: sft` pages may be used to synthesize fine-tuning data. `role: test` pages are held out
from all fine-tuning data but remain in the retrieval index, otherwise the evaluation would be
unanswerable by construction.

## Scope decisions

- `tasks/text_generation.md` does not exist at `{docset['tag']}`; `tasks/language_modeling.md`
  is used as the text-generation task page.
- `main_classes/model.md` and `main_classes/trainer.md` were excluded as stubs (under 3 KB).
- `generation_features.md`, `chat_templating.md` and `serialization.md` were excluded to keep
  three topics with minimal overlap.

## Model weights

The documentation license does not cover model weights. The base model
(`Qwen/Qwen2.5-1.5B-Instruct`) is published under Apache-2.0; the fallback
(`Qwen/Qwen3-4B-Instruct-2507`) is also Apache-2.0. Verify the model card before redistributing
any merged or fine-tuned weights.
"""


def fetch(args: argparse.Namespace) -> int:
    config = load_config()
    docs = config["docs"]
    pages_cfg = config["pages"]
    validate_pages(pages_cfg)

    repo = docs["repo"]
    tag = args.tag or docs["tag"]
    source_dir = docs["source_dir"]
    commit = resolve_commit(repo, tag)

    previous = read_json(DOCSET_PATH)
    pinned_before = previous and previous.get("repo") == repo and previous.get("tag") == tag
    if pinned_before and previous.get("commit_sha") == commit:
        retrieved_at = previous["retrieved_at"]
    else:
        retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    license_url = f"https://github.com/{repo}/blob/{commit}/{docs['license_path']}"
    license_text = http_get(
        f"https://raw.githubusercontent.com/{repo}/{commit}/{docs['license_path']}"
    ).decode("utf-8")
    for marker in REQUIRED_LICENSE_MARKERS:
        if marker not in license_text:
            raise RuntimeError(f"license text is missing expected marker {marker!r}")

    MD_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for page in pages_cfg:
        raw_url = f"https://raw.githubusercontent.com/{repo}/{commit}/{source_dir}/{page['path']}"
        data = http_get(raw_url)
        text = data.decode("utf-8")
        title, headings = parse_markdown(text)
        if not title:
            raise RuntimeError(f"{page['id']}: no H1 title found at {raw_url}")
        if page.get("title") and title != page["title"]:
            raise RuntimeError(
                f"{page['id']}: expected title {page['title']!r} but page says {title!r}"
            )
        if not headings:
            raise RuntimeError(f"{page['id']}: no H2/H3 headings found")

        raw_path = MD_DIR / f"{page['id']}.md"
        raw_path.write_bytes(data)

        docs_url = f"{docs['docs_site'].format(tag=tag)}/{page['path'][:-3]}"
        blob_url = f"{docs['blob_base'].format(repo=repo, rev=commit, source_dir=source_dir)}/{page['path']}"
        manifest.append(
            {
                "page_id": page["id"],
                "title": title,
                "role": page["role"],
                "topics": page["topics"],
                "source_path": f"{source_dir}/{page['path']}",
                "raw_path": str(raw_path.relative_to(ROOT)),
                "url": blob_url,
                "docs_url": docs_url,
                "sha256": sha256_text(text),
                "bytes": len(data),
                "chars": len(text),
                "headings": headings,
                "license": docs["license"],
                "license_url": license_url,
                "repo": repo,
                "tag": tag,
                "commit_sha": commit,
                "retrieved_at": retrieved_at,
            }
        )

    docset = {
        "repo": repo,
        "tag": tag,
        "commit_sha": commit,
        "source_dir": source_dir,
        "license": docs["license"],
        "license_url": license_url,
        "license_sha256": sha256_text(license_text),
        "retrieved_at": retrieved_at,
        "page_count": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "total_chars": sum(item["chars"] for item in manifest),
        "pages": [
            {
                "page_id": item["page_id"],
                "path": pages_cfg[index]["path"],
                "title": item["title"],
                "role": item["role"],
                "topics": item["topics"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for index, item in enumerate(manifest)
        ],
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    DOCSET_PATH.write_text(json.dumps(docset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LICENSE_DOC.parent.mkdir(parents=True, exist_ok=True)
    LICENSE_DOC.write_text(license_document(docset), encoding="utf-8")

    print(f"pinned {repo}@{tag} -> {commit}")
    print(f"{'page':26} {'role':5} {'bytes':>7} {'h2/h3':>6}  title")
    for item in manifest:
        print(
            f"{item['page_id']:26} {item['role']:5} {item['bytes']:>7,} "
            f"{len(item['headings']):>6}  {item['title']}"
        )
    print(
        f"total: {docset['page_count']} pages, {docset['total_bytes']:,} bytes, "
        f"~{docset['total_chars'] // 4:,} tokens"
    )
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}, {DOCSET_PATH.relative_to(ROOT)}, "
          f"{LICENSE_DOC.relative_to(ROOT)}")
    return 0


def check() -> int:
    docset = read_json(DOCSET_PATH)
    manifest = read_manifest()
    if not docset:
        print(f"FAIL: {DOCSET_PATH} is missing; run `make fetch`", file=sys.stderr)
        return 1
    if not manifest:
        print(f"FAIL: {MANIFEST_PATH} is missing; run `make fetch`", file=sys.stderr)
        return 1
    if docset["page_count"] != len(manifest):
        print(
            f"FAIL: docset lists {docset['page_count']} pages, manifest has {len(manifest)}",
            file=sys.stderr,
        )
        return 1

    failures = 0
    for item in manifest:
        path = ROOT / item["raw_path"]
        if not path.exists():
            print(f"FAIL {item['page_id']}: missing {item['raw_path']}", file=sys.stderr)
            failures += 1
            continue
        data = path.read_bytes()
        actual = sha256_text(data.decode("utf-8"))
        if actual != item["sha256"]:
            print(f"FAIL {item['page_id']}: sha256 mismatch", file=sys.stderr)
            failures += 1
        if len(data) != item["bytes"]:
            print(f"FAIL {item['page_id']}: byte count mismatch", file=sys.stderr)
            failures += 1
        title, headings = parse_markdown(data.decode("utf-8"))
        if title != item["title"]:
            print(f"FAIL {item['page_id']}: title drifted ({title!r})", file=sys.stderr)
            failures += 1
        if [h["anchor"] for h in headings] != [h["anchor"] for h in item["headings"]]:
            print(f"FAIL {item['page_id']}: anchors drifted", file=sys.stderr)
            failures += 1
        if item["license"] != docset["license"]:
            print(f"FAIL {item['page_id']}: license mismatch", file=sys.stderr)
            failures += 1

    if failures:
        print(f"{failures} provenance check(s) failed", file=sys.stderr)
        return 1

    roles = {}
    for item in manifest:
        roles[item["role"]] = roles.get(item["role"], 0) + 1
    print(
        f"OK: {docset['repo']}@{docset['tag']} ({docset['commit_sha'][:12]}), "
        f"{len(manifest)} pages, {docset['total_bytes']:,} bytes, "
        f"{roles.get('sft', 0)} sft / {roles.get('test', 0)} test, {docset['license']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="override the pinned tag from configs/project.yaml")
    parser.add_argument("--check", action="store_true", help="verify local copies offline")
    parser.add_argument("--force", action="store_true", help="re-download even when pinned")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    return fetch(args)


if __name__ == "__main__":
    sys.exit(main())
