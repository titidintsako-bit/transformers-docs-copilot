import csv
import json
import unittest

from src.common import CHUNKS_PATH, LABEL_SHEET_PATH, QUESTIONS_PATH, SPLIT_PATH, read_jsonl
from src.eval_questions import build_split, validate


class EvaluationSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QUESTIONS_PATH.exists() or not CHUNKS_PATH.exists():
            raise unittest.SkipTest("run `make chunk` and generate the question set first")
        cls.questions = read_jsonl(QUESTIONS_PATH)
        cls.chunks = read_jsonl(CHUNKS_PATH)
        cls.quotas = {
            "total": 55,
            "dev": 12,
            "test": 43,
            "answerable": 40,
            "unanswerable": 10,
            "parametric": 5,
            "min_per_test_page": 5,
            "min_per_dev_page": 1,
        }

    def test_validation_passes(self):
        self.assertEqual(validate(self.questions, self.chunks, self.quotas), [])

    def test_counts(self):
        self.assertEqual(len(self.questions), 55)
        categories = {}
        splits = {}
        for question in self.questions:
            categories[question["category"]] = categories.get(question["category"], 0) + 1
            splits[question["split"]] = splits.get(question["split"], 0) + 1
        self.assertEqual(categories, {"answerable": 40, "unanswerable": 10, "parametric": 5})
        self.assertEqual(splits, {"dev": 12, "test": 43})

    def test_split_is_page_disjoint(self):
        dev_pages = {
            target["page_id"]
            for question in self.questions
            if question["split"] == "dev"
            for target in question["gold"]
        }
        test_pages = {
            target["page_id"]
            for question in self.questions
            if question["split"] == "test"
            for target in question["gold"]
        }
        self.assertEqual(dev_pages & test_pages, set())

    def test_abstain_questions_have_no_gold(self):
        for question in self.questions:
            if question["category"] != "answerable":
                self.assertEqual(question["gold"], [], question["qid"])
                self.assertEqual(question["expected"], "abstain")
                self.assertEqual(question["split"], "test")

    def test_every_answerable_has_reference_and_gold(self):
        for question in self.questions:
            if question["category"] == "answerable":
                self.assertTrue(question["gold"], question["qid"])
                self.assertTrue(question["reference"].strip(), question["qid"])
                self.assertEqual(question["expected"], "answer")

    def test_split_manifest_matches(self):
        if not SPLIT_PATH.exists():
            self.skipTest("run `make validate-eval` first")
        stored = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored, build_split(self.questions, self.chunks))

    def test_label_sheet_shape(self):
        if not LABEL_SHEET_PATH.exists():
            self.skipTest("run `make label-sheet` first")
        with LABEL_SHEET_PATH.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 55)
        self.assertEqual(len(rows[0]), 15)
        self.assertEqual({row["qid"] for row in rows}, {q["qid"] for q in self.questions})
        for row in rows:
            self.assertTrue(row["reference"].strip(), row["qid"])

    def test_out_of_scope_quota(self):
        out = [q for q in self.questions if q["topic"] == "out-of-scope"]
        self.assertEqual(len(out), 15)
        self.assertTrue(all(q["category"] != "answerable" for q in out))
