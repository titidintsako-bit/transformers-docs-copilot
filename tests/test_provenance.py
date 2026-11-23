import json
import unittest
from pathlib import Path

from src.text import parse_markdown, slugify

ROOT = Path(__file__).resolve().parents[1]
DOCSET_PATH = ROOT / "data" / "raw" / "docset.json"
MANIFEST_PATH = ROOT / "data" / "raw" / "manifest.jsonl"
DATA_LICENSE = ROOT / "docs" / "DATA-LICENSE.md"


def load_manifest():
    return [json.loads(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line]


class SlugifyTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Loading models"), "loading-models")

    def test_punctuation_and_case(self):
        self.assertEqual(slugify("What's new? (v5)"), "whats-new-v5")

    def test_html_comments_ignored(self):
        text = "<!-- Copyright 2024 -->\n# Title\n## Section\n"
        title, headings = parse_markdown(text)
        self.assertEqual(title, "Title")
        self.assertEqual([h["anchor"] for h in headings], ["section"])

    def test_fenced_code_headings_ignored(self):
        text = "# T\n\n```python\n# not a heading\n```\n\n## Real\n"
        _, headings = parse_markdown(text)
        self.assertEqual([h["text"] for h in headings], ["Real"])

    def test_duplicate_anchors_suffixed(self):
        _, headings = parse_markdown("# T\n\n## Example\n\n## Example\n")
        self.assertEqual([h["anchor"] for h in headings], ["example", "example-1"])


class ProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DOCSET_PATH.exists() or not MANIFEST_PATH.exists():
            raise unittest.SkipTest("run `make fetch` first")
        cls.docset = json.loads(DOCSET_PATH.read_text(encoding="utf-8"))
        cls.manifest = load_manifest()

    def test_manifest_matches_docset(self):
        self.assertEqual(self.docset["page_count"], len(self.manifest))
        self.assertEqual(
            [p["page_id"] for p in self.docset["pages"]],
            [m["page_id"] for m in self.manifest],
        )
        self.assertEqual(
            sum(m["bytes"] for m in self.manifest), self.docset["total_bytes"]
        )

    def test_raw_files_are_pinned(self):
        for item in self.manifest:
            with self.subTest(page=item["page_id"]):
                path = ROOT / item["raw_path"]
                self.assertTrue(path.exists(), item["raw_path"])
                raw = path.read_bytes()
                self.assertEqual(len(raw), item["bytes"])
                from src.text import sha256_text

                self.assertEqual(sha256_text(raw.decode("utf-8")), item["sha256"])

    def test_titles_match_pages(self):
        for item in self.manifest:
            with self.subTest(page=item["page_id"]):
                title, headings = parse_markdown((ROOT / item["raw_path"]).read_text(encoding="utf-8"))
                self.assertEqual(title, item["title"])
                self.assertEqual([h["anchor"] for h in headings], [h["anchor"] for h in item["headings"]])
                self.assertTrue(headings, "page has no H2/H3 headings")

    def test_roles_are_disjoint_and_valid(self):
        roles = {item["page_id"]: item["role"] for item in self.manifest}
        self.assertTrue(roles)
        self.assertTrue(set(roles.values()) <= {"sft", "test"})
        sft = {p for p, r in roles.items() if r == "sft"}
        test = {p for p, r in roles.items() if r == "test"}
        self.assertTrue(sft and test)
        self.assertFalse(sft & test)
        for item in self.manifest:
            self.assertTrue(item["topics"], item["page_id"])

    def test_license_provenance(self):
        self.assertEqual(self.docset["license"], "Apache-2.0")
        self.assertTrue(self.docset["license_url"].endswith("/LICENSE"))
        self.assertIn(self.docset["commit_sha"], self.docset["license_url"])
        for item in self.manifest:
            self.assertEqual(item["license"], "Apache-2.0")
            self.assertEqual(item["commit_sha"], self.docset["commit_sha"])
            self.assertIn(item["commit_sha"], item["url"])
            self.assertIn(self.docset["tag"], item["docs_url"])

    def test_commit_is_full_sha(self):
        self.assertRegex(self.docset["commit_sha"], r"^[0-9a-f]{40}$")

    def test_data_license_document(self):
        text = DATA_LICENSE.read_text(encoding="utf-8")
        self.assertIn(self.docset["commit_sha"], text)
        self.assertIn("Apache License, Version 2.0", text)
        self.assertIn(self.docset["tag"], text)
        for item in self.manifest:
            self.assertIn(item["sha256"][:12], text)

    def test_unique_anchors_per_page(self):
        for item in self.manifest:
            with self.subTest(page=item["page_id"]):
                anchors = [h["anchor"] for h in item["headings"]]
                self.assertEqual(len(anchors), len(set(anchors)))


if __name__ == "__main__":
    unittest.main()
