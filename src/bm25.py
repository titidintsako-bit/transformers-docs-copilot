"""A small BM25 implementation: deterministic, dependency-free, inspectable.

Scoring: Okapi BM25 with k1=1.5, b=0.75. Query terms are deduplicated. Documents
with a zero score are never returned, so a miss stays a miss instead of being
papered over by an arbitrary zero-score chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset(
    """a an and are as at be been being but by can could did do does doing for from had has
    have having he her here hers him his how i if in into is it its itself just me more most
    my no nor not now of off on once only or other our out over own same she should so some
    such than that the their them then there these they this those through to too under until
    up very was we were what when where which while who whom why will with would you your""".split()
)

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with camelCase and snake_case split into parts.

    `AutoModelForCausalLM` -> auto, model, causal, lm
    `pre-trained`          -> pre, trained
    """
    text = _CAMEL.sub(" ", text)
    text = text.replace("_", " ")
    return [token for token in _WORD.findall(text.lower()) if token not in STOPWORDS]


class BM25:
    def __init__(self, texts: list[str], ids: list[str], k1: float = 1.5, b: float = 0.75):
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length")
        if not texts:
            raise ValueError("cannot build an index with no documents")
        self.k1 = k1
        self.b = b
        self.ids = list(ids)
        self.doc_tokens = [tokenize(text) for text in texts]
        self.tf = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.n = len(self.doc_tokens)
        self.avgdl = sum(self.doc_len) / self.n
        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def _score(self, query_terms: set[str], index: int) -> float:
        tf = self.tf[index]
        dl = self.doc_len[index]
        normalizer = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        score = 0.0
        for term in query_terms:
            frequency = tf.get(term, 0)
            if frequency:
                score += self.idf[term] * frequency * (self.k1 + 1) / (frequency + normalizer)
        return score

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        query_terms = set(tokenize(query))
        if not query_terms or k <= 0:
            return []
        scored = [
            (self._score(query_terms, index), index)
            for index in range(self.n)
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(self.ids[index], score) for score, index in scored[:k]]

    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "ids": self.ids, "doc_tokens": self.doc_tokens}

    @classmethod
    def from_dict(cls, payload: dict) -> "BM25":
        model = cls.__new__(cls)
        model.k1 = float(payload["k1"])
        model.b = float(payload["b"])
        model.ids = list(payload["ids"])
        model.doc_tokens = [list(tokens) for tokens in payload["doc_tokens"]]
        model.tf = [Counter(tokens) for tokens in model.doc_tokens]
        model.doc_len = [len(tokens) for tokens in model.doc_tokens]
        model.n = len(model.doc_tokens)
        model.avgdl = sum(model.doc_len) / model.n
        document_frequency: Counter[str] = Counter()
        for tokens in model.doc_tokens:
            document_frequency.update(set(tokens))
        model.idf = {
            term: math.log(1 + (model.n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        return model
