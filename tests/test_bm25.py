import unittest

from src.bm25 import BM25, tokenize

CORPUS = [
    ("a", "Load a pretrained model with AutoModel and from_pretrained."),
    ("b", "Greedy search selects the most likely next token at each step."),
    ("c", "LoRA adapters add a small number of trainable parameters."),
]


def build():
    return BM25([text for _, text in CORPUS], [doc_id for doc_id, _ in CORPUS])


class TokenizerTest(unittest.TestCase):
    def test_camel_case_split(self):
        self.assertEqual(tokenize("AutoModelForCausalLM"), ["auto", "model", "causal", "lm"])

    def test_snake_and_hyphen_split(self):
        self.assertEqual(tokenize("pre-trained_model"), ["pre", "trained", "model"])

    def test_stopwords_removed(self):
        self.assertEqual(tokenize("the and of a"), [])
        self.assertEqual(tokenize("How do I load a pretrained model?"), ["load", "pretrained", "model"])

    def test_numbers_kept(self):
        self.assertEqual(tokenize("max_new_tokens=50"), ["max", "new", "tokens", "50"])


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.model = build()

    def test_ranks_matching_document_first(self):
        self.assertEqual(self.model.search("load a pretrained model", k=1)[0][0], "a")
        self.assertEqual(self.model.search("most likely next token", k=1)[0][0], "b")
        self.assertEqual(self.model.search("trainable parameters adapter", k=1)[0][0], "c")

    def test_unknown_terms_return_nothing(self):
        self.assertEqual(self.model.search("zzzzz qqqqq"), [])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.model.search("the and of"), [])

    def test_respects_k(self):
        self.assertEqual(len(self.model.search("model token parameters", k=2)), 2)

    def test_scores_are_sorted_descending(self):
        scores = [score for _, score in self.model.search("model token parameters", k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_deterministic(self):
        self.assertEqual(
            self.model.search("model token parameters", k=3),
            self.model.search("model token parameters", k=3),
        )

    def test_roundtrip(self):
        clone = BM25.from_dict(self.model.to_dict())
        self.assertEqual(
            clone.search("trainable parameters", k=3),
            self.model.search("trainable parameters", k=3),
        )
