import tempfile
import unittest
from pathlib import Path

from src.common import load_config
from src.train_qlora import CausalCollator, build_tokenized, tokenize_example, training_arguments


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT:" + messages[-1]["content"]

    def __call__(self, text, add_special_tokens=False):
        if text.startswith("PROMPT:"):
            return {"input_ids": [1, 2, 3]}
        return {"input_ids": [4, 5]}


EXAMPLE = {
    "id": "x",
    "messages": [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "Answer: supported [1]."},
    ],
}


class TokenizeTest(unittest.TestCase):
    def test_masks_prompt_and_supervises_answer(self):
        item = tokenize_example(EXAMPLE, FakeTokenizer(), max_length=64)
        self.assertEqual(item["input_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(item["attention_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(item["labels"], [-100, -100, -100, 4, 5])

    def test_returns_none_when_too_long(self):
        self.assertIsNone(tokenize_example(EXAMPLE, FakeTokenizer(), max_length=4))

    def test_missing_assistant_target_raises(self):
        with self.assertRaises(ValueError):
            tokenize_example({"id": "bad", "messages": [{"role": "user", "content": "q"}]}, FakeTokenizer(), 64)

    def test_build_tokenized_counts_skips(self):
        rows = [EXAMPLE, {**EXAMPLE, "id": "long"}]
        items, skipped = build_tokenized(rows, FakeTokenizer(), max_length=64)
        self.assertEqual((len(items), skipped), (2, 0))


class TrainingArgumentsTest(unittest.TestCase):
    def test_supported_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = training_arguments(
                output_dir=Path(tmp), epochs=2, config=load_config(), bf16=False
            )
        self.assertEqual(args.num_train_epochs, 2)
        self.assertEqual(args.per_device_train_batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.warmup_steps, 2)


class CollatorTest(unittest.TestCase):
    def test_right_pads_both_labels_and_attention(self):
        collator = CausalCollator(pad_token_id=0)
        batch = collator(
            [
                {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                {"input_ids": [3], "attention_mask": [1], "labels": [3]},
            ]
        )
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1], [1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 2], [3, -100]])


if __name__ == "__main__":
    unittest.main()
