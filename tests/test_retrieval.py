import json
import unittest

from src.chunk import build_chunks
from src.common import PROBES_PATH, docset, read_manifest
from src.eval_retrieval import (
    aggregate,
    evaluate,
    is_relevant,
    validate_probes,
)
from src.retrieve import Retriever


class RetrievalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = read_manifest()
        if not manifest:
            raise unittest.SkipTest("run `make fetch` first")
        info = docset()
        cls.chunks = build_chunks(
            manifest, tag=info["tag"], max_tokens=550, overlap=0.15
        )
        cls.retriever = Retriever.from_chunks(cls.chunks)
        cls.ks = (1, 3, 5)

    def test_sanity_queries(self):
        cases = {
            "what is greedy search": ("generation_strategies", "greedy-search"),
            "ELI5 dataset": ("tasks_language_modeling", None),
            "LoRA adapter": ("peft", None),
        }
        for query, (page, anchor) in cases.items():
            with self.subTest(query=query):
                top = self.retriever.search(query, k=1)[0]
                self.assertEqual(top["page_id"], page)
                if anchor:
                    self.assertEqual(anchor, top["anchor"])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.retriever.search("the and of", k=5), [])

    def test_results_carry_metadata(self):
        for result in self.retriever.search("generation configuration", k=5):
            self.assertIn("score", result)
            self.assertIn("chunk_id", result)
            self.assertIn("url", result)
            self.assertGreater(result["score"], 0)

    def test_page_filter(self):
        results = self.retriever.search("training", k=5, page_id="peft")
        self.assertTrue(results)
        self.assertTrue(all(result["page_id"] == "peft" for result in results))

    def test_probe_validation(self):
        probes = [json.loads(line) for line in PROBES_PATH.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(validate_probes(probes, self.chunks), [])

    def test_probe_quality_floor(self):
        probes = [json.loads(line) for line in PROBES_PATH.read_text(encoding="utf-8").splitlines() if line]
        rows = evaluate(self.retriever, probes, self.ks)
        overall = aggregate(rows, self.ks)
        self.assertGreaterEqual(overall["recall@5"], 0.9)
        self.assertGreaterEqual(overall["recall@1"], 0.6)
        self.assertGreaterEqual(overall["mrr@5"], 0.7)

    def test_relevance_rules(self):
        chunk = {
            "page_id": "models",
            "anchor": "sharded-checkpoints",
            "section_anchors": ["large-models", "sharded-checkpoints"],
        }
        self.assertTrue(is_relevant(chunk, [{"page_id": "models", "anchor": None}]))
        self.assertTrue(is_relevant(chunk, [{"page_id": "models", "anchor": "large-models"}]))
        self.assertTrue(
            is_relevant(chunk, [{"page_id": "models", "anchor": "sharded-checkpoints"}])
        )
        self.assertFalse(is_relevant(chunk, [{"page_id": "models", "anchor": "custom-models"}]))
        self.assertFalse(is_relevant(chunk, [{"page_id": "peft", "anchor": None}]))
