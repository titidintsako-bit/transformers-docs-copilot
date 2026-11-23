import unittest

from src.generate import ABSTAIN_TEXT, build_messages, parse_response

BASE = "https://github.com/huggingface/transformers/blob/abc123/docs/source/en"


def passage(page_id="models", anchor=None, title="Loading models", section="Loading models", n=1):
    url = f"{BASE}/{page_id}.md" + (f"#{anchor}" if anchor else "")
    return {
        "chunk_id": f"{page_id}:{anchor or '_intro'}:0",
        "page_id": page_id,
        "page_title": title,
        "section": section,
        "anchor": anchor,
        "url": url,
        "text": f"Transformers v5.24.0 > {section}\n\nPassage {n} body text about models.",
    }


class PromptTest(unittest.TestCase):
    def test_numbered_passages_and_urls(self):
        messages = build_messages("How do I load a model?", [passage(), passage("peft", "add-an-adapter", "PEFT", "Add an adapter", 2)])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(ABSTAIN_TEXT, messages[0]["content"])
        user = messages[1]["content"]
        self.assertIn("[1] Loading models", user)
        self.assertIn("[2] PEFT", user)
        self.assertIn(f"{BASE}/models.md", user)
        self.assertIn(f"{BASE}/peft.md#add-an-adapter", user)
        self.assertTrue(user.rstrip().endswith("How do I load a model?"))

    def test_no_passages_stated(self):
        messages = build_messages("Unanswerable?", [])
        self.assertIn("No passages were retrieved", messages[1]["content"])

    def test_gold_labels_never_in_prompt(self):
        messages = build_messages("q", [passage()])
        joined = " ".join(message["content"] for message in messages)
        self.assertNotIn("gold", joined.lower())


class ParseTest(unittest.TestCase):
    def setUp(self):
        self.passages = [passage(), passage("peft", "add-an-adapter", "PEFT", "Add an adapter", 2)]

    def test_valid_answer(self):
        raw = (
            "Answer: Call from_pretrained to load the weights [1]. "
            "Adapters can be attached separately [2].\n"
            f"Sources:\n[1] Loading models - {BASE}/models.md\n"
            f"[2] PEFT - {BASE}/peft.md#add-an-adapter"
        )
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "answered")
        self.assertEqual(parsed["citations"], [1, 2])
        self.assertTrue(parsed["format_ok"])
        self.assertEqual(parsed["repairs"], [])
        self.assertEqual(parsed["sources"][1]["anchor"], "add-an-adapter")
        self.assertIn("[1]", parsed["answer"])

    def test_missing_source_line_is_repaired(self):
        raw = "Answer: Call from_pretrained [1].\nSources:\n"
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "answered")
        self.assertEqual(parsed["citations"], [1])
        self.assertEqual(parsed["repairs"], [{"action": "added_missing_source", "n": 1}])
        self.assertFalse(parsed["format_ok"])

    def test_out_of_range_citation_forces_abstention(self):
        raw = "Answer: Use the flag [9].\nSources:\n[9] Nope - https://example.com/x"
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "forced_abstain")
        self.assertEqual(parsed["reason"], "no_valid_citations")
        self.assertEqual(parsed["answer"], ABSTAIN_TEXT)
        self.assertTrue(any(repair["action"] == "removed_out_of_range_citation" for repair in parsed["repairs"]))

    def test_url_mismatch_is_repaired_and_flagged(self):
        raw = (
            "Answer: Use from_pretrained [1].\n"
            f"Sources:\n[1] Loading models - {BASE}/peft.md"
        )
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "answered")
        self.assertEqual(parsed["sources"][0]["url"], f"{BASE}/models.md")
        self.assertEqual(parsed["invalid_source_entries"][0]["reason"], "url_mismatch")
        self.assertIn({"action": "replaced_bad_source_url", "n": 1}, parsed["repairs"])
        self.assertFalse(parsed["format_ok"])

    def test_invented_url_in_answer_forces_abstention(self):
        raw = (
            "Answer: See https://example.com/fake for details [1].\n"
            f"Sources:\n[1] Loading models - {BASE}/models.md"
        )
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "forced_abstain")
        self.assertEqual(parsed["reason"], "unverified_url_in_answer")
        self.assertEqual(parsed["unverified_urls"], ["https://example.com/fake"])

    def test_exact_abstention(self):
        parsed = parse_response(ABSTAIN_TEXT, self.passages)
        self.assertEqual(parsed["status"], "abstained")
        self.assertTrue(parsed["format_ok"])
        self.assertEqual(parsed["citations"], [])

    def test_abstention_after_answer_prefix(self):
        parsed = parse_response(f"Answer: {ABSTAIN_TEXT}", self.passages)
        self.assertEqual(parsed["status"], "abstained")

    def test_abstention_with_sources_is_format_failure(self):
        raw = f"{ABSTAIN_TEXT}\nSources:\n[1] Loading models - {BASE}/models.md"
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "abstained")
        self.assertFalse(parsed["format_ok"])

    def test_url_without_anchor_is_accepted_but_flagged(self):
        raw = (
            "Answer: Adapters attach separately [2].\n"
            f"Sources:\n[2] PEFT - {BASE}/peft.md"
        )
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "answered")
        self.assertFalse(parsed["sources"][0]["anchor_match"])
        self.assertIn({"action": "url_missing_anchor", "n": 2}, parsed["repairs"])
        self.assertFalse(parsed["format_ok"])

    def test_citation_list_expands(self):
        raw = "Answer: Both apply [1, 2].\nSources:\n"
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["citations"], [1, 2])
        self.assertEqual(len(parsed["repairs"]), 2)

    def test_sources_without_inline_citation(self):
        raw = f"Answer: Loading is easy.\nSources:\n[1] Loading models - {BASE}/models.md"
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "answered")
        self.assertFalse(parsed["format_ok"])

    def test_missing_answer_prefix(self):
        raw = f"Call from_pretrained [1].\nSources:\n[1] Loading models - {BASE}/models.md"
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "answered")
        self.assertFalse(parsed["format_ok"])

    def test_empty_output(self):
        parsed = parse_response("", self.passages)
        self.assertEqual(parsed["status"], "forced_abstain")
        self.assertEqual(parsed["reason"], "no_valid_citations")

    def test_unicode_dash_sources(self):
        raw = (
            "Answer: Call from_pretrained [1].\n"
            f"Sources:\n[1] Loading models \u2014 {BASE}/models.md"
        )
        parsed = parse_response(raw, self.passages)
        self.assertEqual(parsed["status"], "answered")
        self.assertEqual(parsed["citations"], [1])


if __name__ == "__main__":
    unittest.main()
