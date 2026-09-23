"""
bm25_search.py
----------------
BM25 keyword search over the same chunks stored in the vector store.
Pure-Python (via rank_bm25), no compiled dependencies — pairs with dense
vector search in hybrid_retriever.py to catch exact keyword/acronym
matches that embeddings alone sometimes miss.
"""

import re

from rank_bm25 import BM25Okapi

# Lightweight, dependency-free suffix stripping so exact-match BM25 isn't
# defeated by simple inflection (e.g. query "review" vs. document
# "reviewed", "policy" vs "policies"). Not a real stemmer, but enough to
# stop common word-form mismatches from hiding otherwise-relevant chunks.
_SUFFIXES = ("ational", "edness", "ement", "tional", "ing", "edly", "ies", "ment", "ness", "ed", "es", "ly", "s")


def _stem(word: str) -> str:
    if len(word) <= 4:
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"\b\w+\b", text.lower())]


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]):
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int) -> list[dict]:
        if self.bm25 is None or not self.chunks:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_idx[:top_k]:
            if scores[idx] <= 0:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results
