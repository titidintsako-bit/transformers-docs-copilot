import unittest

from src.build_sft import (
    NEGATIVE_QUESTIONS,
    SEED_QUESTIONS,
    build_examples,
    clean_prose,
    make_target,
    question_variants,
    select_sentences,
    split_examples,
)
from src.common import CHUNKS_PATH, QUESTIONS_PATH, read_jsonl
from src.generate import ABSTAIN_TEXT
from src.retrieve import Retriever

BASE = "https://github.com/huggingface/transformers/blob/abc123/docs/source/en"


def chunk(chunk_id, role, text, section="Sky", page="Weather"):
    return {
        "chunk_id": chunk_id,
        "page_id": page.lower(),
        "page_title": page,
        "role": role,
        "section": section,
        "anchor": section.lower(),
        "url": f"{BASE}/{page.lower()}.md",
        "text": f"Transformers v5.24.0 > {section}\n\n{text}",
    }


class CleanTest(unittest.TestCase):
    def test_removes_urls_links_code_tables_and_admonitions(self):
        text = (
            "Transformers v5.24.0 > Sky\n\n"
            "> [!NOTE]\n> The sky appears blue because of Rayleigh scattering.\n\n"
            "```py\nprint('hidden')\n```\n"
            "See [the guide](https://example.com/guide) and https://example.com/raw.\n"
            "| a | b |\n|---|---|\n| 1 | 2 |"
        )
        cleaned = clean_prose(text)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("hidden", cleaned)
        self.assertNotIn("[!NOTE]", cleaned)
        self.assertIn("the guide", cleaned)
        self.assertIn("Rayleigh", cleaned)


class TargetTest(unittest.TestCase):
    def test_selects_relevant_sentence(self):
        text = "The grass is green. The sky appears blue because of Rayleigh scattering. Cows eat grass."
        selected = select_sentences(text, "Why is the sky blue?")
        self.assertTrue(any("sky" in sentence for sentence in selected))
        self.assertFalse(any("Cows" in sentence for sentence in selected))

    def test_target_has_prefix_and_inline_citation(self):
        target = make_target(["The sky appears blue because of Rayleigh scattering."], 2)
        self.assertEqual(
            target,
            "Answer: The sky appears blue because of Rayleigh scattering [2].",
        )

    def test_question_variants_prefer_seed(self):
        seed_chunk = chunk(
            "quicktour:pipeline:0",
            "sft",
            "The Pipeline class is the most convenient way to run inference with a pretrained model.",
        )
        variants = question_variants(seed_chunk)
        self.assertEqual(len(variants), 3)
        self.assertEqual(variants[0], SEED_QUESTIONS["quicktour:pipeline:0"])


class BuildTest(unittest.TestCase):
    def test_only_sft_chunks_become_context(self):
        sft = chunk(
            "weather:sky:0",
            "sft",
            "The sky appears blue because of Rayleigh scattering. Clear skies are common in summer.",
        )
        held_out = chunk(
            "secret:sky:0",
            "test",
            "SECRET held-out page text that must never enter training.",
        )
        retriever = Retriever.from_chunks([sft])
        examples = build_examples([sft, held_out], retriever, negatives=["What is the capital of France?"])
        self.assertTrue(examples)
        for example in examples:
            joined = " ".join(message["content"] for message in example["messages"])
            self.assertNotIn("SECRET", joined)
            if example["category"] == "answerable":
                self.assertEqual(example["source_chunk"], "weather:sky:0")
                self.assertEqual(example["citation"], 1)
                self.assertRegex(example["target"], r"\[1\]")
            else:
                self.assertEqual(example["target"], ABSTAIN_TEXT)

    def test_split_has_no_group_leakage(self):
        examples = [
            {"id": f"p{i}", "group": f"positive:{i}", "category": "answerable"}
            for i in range(10)
        ] + [
            {"id": f"n{i}", "group": f"negative:{i}", "category": "abstain"}
            for i in range(10)
        ]
        train, val = split_examples(examples)
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertFalse({row["group"] for row in train} & {row["group"] for row in val})

    def test_real_data_does_not_use_test_pages_or_eval_questions(self):
        chunks = read_jsonl(CHUNKS_PATH)
        sft_chunks = [row for row in chunks if row["role"] == "sft"]
        retriever = Retriever.from_chunks(sft_chunks)
        examples = build_examples(chunks, retriever)
        sft_ids = {row["chunk_id"] for row in sft_chunks}
        eval_questions = {row["question"] for row in read_jsonl(QUESTIONS_PATH)}

        self.assertTrue(examples)
        self.assertFalse({row["question"] for row in examples} & eval_questions)
        for example in examples:
            if example["source_chunk"]:
                self.assertIn(example["source_chunk"], sft_ids)
            for message in example["messages"]:
                if message["role"] == "assistant":
                    self.assertNotIn("https://", message["content"])
        self.assertIn("models:sharded-checkpoints:0", SEED_QUESTIONS)
        self.assertGreater(len(NEGATIVE_QUESTIONS), 40)


if __name__ == "__main__":
    unittest.main()
