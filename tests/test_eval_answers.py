import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.eval_answers import run
from src.generate import ABSTAIN_TEXT, parse_response
from src.retrieve import Retriever

BASE = "https://github.com/huggingface/transformers/blob/abc123/docs/source/en"


def chunk(chunk_id, page_id, anchor, text, title="Doc", section=None):
    url = f"{BASE}/{page_id}.md" + (f"#{anchor}" if anchor else "")
    return {
        "chunk_id": chunk_id,
        "page_id": page_id,
        "page_title": title,
        "section": section or anchor or "intro",
        "anchor": anchor,
        "section_anchors": [anchor] if anchor else [],
        "url": url,
        "text": f"Transformers v5.24.0 > {section or anchor or 'intro'}\n\n{text}",
        "role": "test",
    }


CHUNKS = [
    chunk("doc:sec:0", "doc", "sec", "Zebra decoding uses stripes by default.", "Doc", "Section"),
    chunk("other:sec:0", "other", "sec", "Unrelated material about llamas.", "Other", "Section"),
]
QUESTIONS = [
    {
        "qid": "a-01",
        "split": "test",
        "topic": "generation",
        "category": "answerable",
        "answerable": True,
        "expected": "answer",
        "question": "zebra decoding default?",
        "gold": [{"page_id": "doc", "anchor": "sec"}],
        "reference": "stripes",
    },
    {
        "qid": "a-02",
        "split": "test",
        "topic": "out-of-scope",
        "category": "unanswerable",
        "answerable": False,
        "expected": "abstain",
        "question": "zebra question that is out of scope?",
        "gold": [],
        "reference": "not in docs",
    },
    {
        "qid": "a-03",
        "split": "test",
        "topic": "loading",
        "category": "answerable",
        "answerable": True,
        "expected": "answer",
        "question": "unicornxyz not in the corpus?",
        "gold": [{"page_id": "other", "anchor": "sec"}],
        "reference": "llamas",
    },
    {
        "qid": "a-04",
        "split": "test",
        "topic": "generation",
        "category": "answerable",
        "answerable": True,
        "expected": "answer",
        "question": "zebra decoding invented url?",
        "gold": [{"page_id": "doc", "anchor": "sec"}],
        "reference": "stripes",
    },
]


def fake_generator(question, passages):
    url = passages[0]["url"]
    if "out of scope" in question:
        return ABSTAIN_TEXT
    if "invented url" in question:
        return f"Answer: Zebra uses stripes https://example.com/madeup [1].\nSources:\n[1] Doc - {url}"
    return f"Answer: Zebra decoding uses stripes by default [1].\nSources:\n[1] Doc - {url}"


class EvalAnswersTest(unittest.TestCase):
    def setUp(self):
        self.retriever = Retriever.from_chunks(CHUNKS)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_eval(self, generator=fake_generator, questions=None):
        return run(
            self.retriever,
            questions or QUESTIONS,
            generator,
            k=2,
            split="test",
            limit=0,
            out_prefix=self.root / "answers_baseline",
            label_out=self.root / "label_sheet_baseline.csv",
            model_id="fake/model",
            quantization="none",
            max_new_tokens=64,
        )

    def test_scores_contract_and_behaviour(self):
        payload = self.run_eval()
        rows = {row["qid"]: row for row in payload["rows"]}
        self.assertEqual(payload["n"], 4)

        self.assertEqual(rows["a-01"]["status"], "answered")
        self.assertTrue(rows["a-01"]["behavior_ok"])
        self.assertTrue(rows["a-01"]["citation_hit"])
        self.assertTrue(rows["a-01"]["format_ok"])

        self.assertEqual(rows["a-02"]["status"], "abstained")
        self.assertTrue(rows["a-02"]["behavior_ok"])
        self.assertIsNone(rows["a-02"]["citation_hit"])

        self.assertEqual(rows["a-03"]["status"], "forced_abstain")
        self.assertEqual(rows["a-03"]["reason"], "no_retrieved_passages")
        self.assertTrue(rows["a-03"]["false_abstain"])
        self.assertFalse(rows["a-03"]["forced_abstain_attempt"])
        self.assertEqual(rows["a-03"]["latency_s"], 0.0)

        self.assertEqual(rows["a-04"]["status"], "forced_abstain")
        self.assertEqual(rows["a-04"]["reason"], "unverified_url_in_answer")
        self.assertEqual(rows["a-04"]["answer"], ABSTAIN_TEXT)
        self.assertTrue(rows["a-04"]["forced_abstain_attempt"])

        overall = payload["overall"]
        self.assertEqual(overall["n"], 4)
        self.assertEqual(overall["behavior_accuracy"], 0.5)
        self.assertEqual(overall["model_abstain_rate_on_negative"], 1.0)
        self.assertEqual(overall["delivered_abstention_rate_on_negative"], 1.0)
        self.assertEqual(overall["over_answer_rate_on_negative"], 0.0)
        self.assertEqual(overall["parser_save_rate_on_negative"], 0.0)
        self.assertEqual(overall["format_rate"], 0.5)

    def test_generator_skipped_without_passages(self):
        calls = []

        def generator(question, passages):
            calls.append(question)
            return fake_generator(question, passages)

        self.run_eval(generator)
        self.assertNotIn("unicornxyz", " ".join(calls))

    def test_generation_error_is_captured(self):
        def broken(question, passages):
            raise RuntimeError("boom")

        payload = self.run_eval(broken, [QUESTIONS[0]])
        row = payload["rows"][0]
        self.assertEqual(row["status"], "forced_abstain")
        self.assertEqual(row["reason"], "generation_error")
        self.assertIn("boom", row["error"])
        self.assertEqual(row["answer"], ABSTAIN_TEXT)

    def test_writes_reports_predictions_and_label_sheet(self):
        self.run_eval()
        json_path = self.root / "answers_baseline.json"
        md_path = self.root / "answers_baseline.md"
        predictions = self.root / "predictions_baseline.jsonl"
        sheet = self.root / "label_sheet_baseline.csv"
        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())
        self.assertTrue(predictions.exists())
        self.assertTrue(sheet.exists())

        payload = json.loads(json_path.read_text())
        self.assertEqual(payload["model_id"], "fake/model")
        self.assertEqual(len(predictions.read_text().strip().splitlines()), 4)
        self.assertIn("Over-answered negative questions", md_path.read_text())

        with sheet.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames
            records = list(reader)
        self.assertIn("supported_0_1_2", header)
        self.assertIn("status", header)
        self.assertIn("notes_auto", header)
        row = next(record for record in records if record["qid"] == "a-01")
        self.assertEqual(row["status"], "answered")
        self.assertEqual(row["retrieval_hit"], "1")
        self.assertEqual(row["abstained"], "0")
        self.assertEqual(row["supported_0_1_2"], "")

    def test_negative_answered_is_over_answer(self):
        payload = self.run_eval(
            lambda question, passages: fake_generator(QUESTIONS[0]["question"], passages),
            [QUESTIONS[1]],
        )
        row = payload["rows"][0]
        self.assertEqual(row["status"], "answered")
        self.assertTrue(row["over_answer"])
        self.assertFalse(row["model_abstained"])
        self.assertFalse(row["behavior_ok"])
        self.assertEqual(payload["overall"]["over_answer_rate_on_negative"], 1.0)
        self.assertEqual(payload["overall"]["parser_save_rate_on_negative"], 0.0)

    def test_negative_rejected_output_is_parser_save(self):
        payload = self.run_eval(lambda question, passages: "Answer: Zebra is great.", [QUESTIONS[1]])
        row = payload["rows"][0]
        self.assertEqual(row["status"], "forced_abstain")
        self.assertTrue(row["forced_abstain_attempt"])
        self.assertFalse(row["over_answer"])
        self.assertFalse(row["model_abstained"])
        self.assertFalse(row["behavior_ok"])
        self.assertEqual(payload["overall"]["parser_save_rate_on_negative"], 1.0)
        self.assertEqual(payload["overall"]["over_answer_rate_on_negative"], 0.0)

    def test_parse_response_rejects_bad_output_before_scoring(self):
        parsed = parse_response("Answer: made up [7]", CHUNKS)
        self.assertEqual(parsed["status"], "forced_abstain")


if __name__ == "__main__":
    unittest.main()
