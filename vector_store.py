"""
vector_store.py
-----------------
Wraps a FAISS index for dense vector search, using sentence-transformers
for embeddings. Supports saving/loading to disk so documents don't need
to be re-embedded on every app restart.
"""

import json
import pickle
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, INDEX_DIR


@lru_cache(maxsize=1)
def _load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: list[dict] = []  # parallel metadata for each vector

    def build(self, chunks: list[dict]):
        """Build a fresh FAISS index from a list of chunk dicts."""
        embedder = _load_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        embeddings = _normalize(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        self.index.add(embeddings.astype("float32"))
        self.chunks = chunks

    def add(self, chunks: list[dict]):
        """Add more chunks to an existing index (e.g. a newly uploaded doc)."""
        if self.index is None:
            self.build(chunks)
            return
        embedder = _load_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        embeddings = _normalize(embeddings)
        self.index.add(embeddings.astype("float32"))
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int) -> list[dict]:
        """Return top_k chunks most similar to the query, each with a 'score' key."""
        if self.index is None or self.index.ntotal == 0:
            return []
        embedder = _load_embedder()
        query_vec = embedder.encode([query], convert_to_numpy=True)
        query_vec = _normalize(query_vec).astype("float32")

        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def save(self, path: str = INDEX_DIR):
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(Path(path) / "index.faiss"))
        with open(Path(path) / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: str = INDEX_DIR) -> bool:
        """Load a previously saved index. Returns True if successful."""
        index_file = Path(path) / "index.faiss"
        chunks_file = Path(path) / "chunks.pkl"
        if not (index_file.exists() and chunks_file.exists()):
            return False
        self.index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        return True

    def clear(self, path: str = INDEX_DIR):
        self.index = None
        self.chunks = []
        index_file = Path(path) / "index.faiss"
        chunks_file = Path(path) / "chunks.pkl"
        index_file.unlink(missing_ok=True)
        chunks_file.unlink(missing_ok=True)

    @property
    def document_names(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms
