import unittest

from src.bm25 import BM25, tokenize
from src.chunk import build_chunks, pack_paragraphs, validate_chunks
from src.common import docset, read_manifest
from src.text import estimate_tokens, iter_paragraphs, split_sections


class ChunkingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_manifest()
        if not cls.manifest:
            raise unittest.SkipTest("run `make fetch` first")
        cls.info = docset()
        cls.max_tokens = 550
        cls.chunks = build_chunks(
            cls.manifest, tag=cls.info["tag"], max_tokens=cls.max_tokens, overlap=0.15
        )

    def test_validation_passes(self):
        self.assertEqual(validate_chunks(self.chunks, self.manifest, self.max_tokens), [])

    def test_every_paragraph_is_indexed(self):
        for page in self.manifest:
            page_chunks = [c for c in self.chunks if c["page_id"] == page["page_id"]]
            self.assertTrue(page_chunks, page["page_id"])
            raw = (self._root() / page["raw_path"]).read_text(encoding="utf-8")
            _, sections = split_sections(raw)
            for section in sections:
                for paragraph, _ in iter_paragraphs(section["body"]):
                    with self.subTest(page=page["page_id"], anchor=section["anchor"]):
                        self.assertTrue(
                            any(paragraph in chunk["text"] for chunk in page_chunks),
                            f"paragraph missing from chunks: {paragraph[:60]!r}",
                        )

    def test_breadcrumb_and_metadata(self):
        for chunk in self.chunks:
            self.assertTrue(chunk["text"].startswith(f"Transformers {self.info['tag']} > "))
            self.assertIn(chunk["page_title"], chunk["breadcrumb"])
            self.assertIn(chunk["chunk_id"], {c["chunk_id"] for c in self.chunks})
            self.assertEqual(chunk["url"], chunk["url"].strip())
            if chunk["anchor"]:
                self.assertTrue(chunk["url"].endswith("#" + chunk["anchor"]))
            self.assertIn(chunk["role"], {"sft", "test"})

    def test_chunk_ids_are_unique_and_ordered(self):
        ids = [c["chunk_id"] for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        for page in self.manifest:
            parts = [c["part"] for c in self.chunks if c["page_id"] == page["page_id"]]
            self.assertEqual(parts[0], 0)

    def test_deterministic(self):
        again = build_chunks(
            self.manifest, tag=self.info["tag"], max_tokens=self.max_tokens, overlap=0.15
        )
        self.assertEqual(self.chunks, again)

    def test_size_bounds(self):
        self.assertTrue(all(c["n_tokens_est"] > 0 for c in self.chunks))
        big = [c for c in self.chunks if c["n_tokens_est"] > self.max_tokens * 1.4]
        self.assertEqual(big, [])

    def _root(self):
        from src.common import ROOT

        return ROOT


class PackingTest(unittest.TestCase):
    def test_packs_under_budget(self):
        paragraphs = ["word " * 100] * 6
        windows = pack_paragraphs(paragraphs, budget=200, overlap=20)
        self.assertGreater(len(windows), 1)
        for window in windows:
            self.assertLessEqual(sum(estimate_tokens(p) for p in window), 200)

    def test_overlap_shares_paragraphs(self):
        paragraphs = ["alpha " * 10, "beta " * 10, "gamma " * 10, "delta " * 10]
        windows = pack_paragraphs(paragraphs, budget=40, overlap=20)
        self.assertGreater(len(windows), 1)
        self.assertTrue(set(windows[0]) & set(windows[1]))

    def test_single_oversize_paragraph_advances(self):
        paragraphs = ["x " * 1000, "y " * 10]
        windows = pack_paragraphs(paragraphs, budget=50, overlap=10)
        self.assertEqual(windows[0], [paragraphs[0]])
        self.assertEqual(windows[-1][-1], paragraphs[1])

    def test_no_empty_windows(self):
        windows = pack_paragraphs(["a", "b", "c"], budget=10, overlap=3)
        self.assertTrue(all(windows))
