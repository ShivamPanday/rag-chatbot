"""
hybrid_retriever.py
---------------------
Combines dense vector search (FAISS) and sparse keyword search (BM25)
using Reciprocal Rank Fusion (RRF), then reranks the fused candidates
with a cross-encoder for higher precision before handing the final
top chunks to the LLM.

Why hybrid + rerank instead of just vector search:
  - Vector search alone misses exact keyword/acronym matches embeddings
    don't represent well (e.g. specific product codes, names)
  - BM25 alone misses semantic paraphrases ("car" vs "automobile")
  - RRF combines both rankings without needing to tune score weights
  - A cross-encoder scores (query, chunk) pairs jointly — much more
    precise than comparing independently-computed embeddings, but too
    slow to run over the whole corpus, so it only reranks the top
    candidates that hybrid search already narrowed down
"""

from functools import lru_cache

from sentence_transformers import CrossEncoder

from bm25_search import BM25Index
from config import (
    BM25_TOP_K,
    FINAL_TOP_K,
    MERGED_TOP_K,
    PER_SOURCE_MIN_CANDIDATES,
    RERANK_TOP_N,
    RERANKER_MODEL,
    RRF_K,
    VECTOR_TOP_K,
)
from vector_store import VectorStore


@lru_cache(maxsize=1)
def _load_reranker():
    return CrossEncoder(RERANKER_MODEL)


def _reciprocal_rank_fusion(
    vector_results: list[dict], bm25_results: list[dict]
) -> list[dict]:
    """Merge two ranked lists into one, scored by RRF."""
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, chunk in enumerate(vector_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_lookup[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_lookup.setdefault(cid, chunk)

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [chunk_lookup[cid] for cid in ranked_ids]


def retrieve(
    query: str, vector_store: VectorStore, bm25_index: BM25Index
) -> list[dict]:
    """
    Full hybrid retrieval pipeline: vector search + BM25 -> RRF fusion
    -> cross-encoder rerank -> final top-k chunks.
    """
    vector_results = vector_store.search(query, VECTOR_TOP_K)
    bm25_results = bm25_index.search(query, BM25_TOP_K)

    fused = _reciprocal_rank_fusion(vector_results, bm25_results)
    if not fused:
        return []

    candidates = _select_candidates_for_rerank(
        fused, RERANK_TOP_N, PER_SOURCE_MIN_CANDIDATES
    )

    reranker = _load_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
    )

    selected = _diversify_by_source(reranked, FINAL_TOP_K)

    final = []
    for chunk, score in selected:
        chunk = dict(chunk)
        chunk["rerank_score"] = float(score)
        final.append(chunk)

    return final


def retrieve_multi(
    sub_queries: list[str], vector_store: VectorStore, bm25_index: BM25Index
) -> list[dict]:
    """
    Run the full hybrid retrieval pipeline independently for each
    sub-query, then merge the results (deduped by chunk, keeping the
    highest score seen), sorted by relevance.

    This is what actually fixes compound/comparison questions ("what's
    the difference between A's policy and B's policy?"). Retrieving with
    a single blended query lets one topic's terms dilute or out-rank the
    other topic's genuinely relevant chunk during reranking — e.g. a
    generic chunk that just repeats a company's name can out-score the
    real content chunk once a second, unrelated topic's terms are mixed
    into the same query. Retrieving each topic on its own keeps that
    topic's search signal clean; merging afterwards gives the answer
    step context for both.

    For a single-topic question, sub_queries has one item and this is
    equivalent to calling retrieve() directly.
    """
    best_by_id: dict[str, dict] = {}
    order: list[str] = []

    for sub_q in sub_queries:
        for chunk in retrieve(sub_q, vector_store, bm25_index):
            cid = chunk["chunk_id"]
            if cid not in best_by_id or chunk["rerank_score"] > best_by_id[cid]["rerank_score"]:
                best_by_id[cid] = chunk
            if cid not in order:
                order.append(cid)

    merged = [best_by_id[cid] for cid in order]
    merged.sort(key=lambda c: c["rerank_score"], reverse=True)
    return merged[:MERGED_TOP_K]


def _select_candidates_for_rerank(
    fused: list[dict], top_n: int, per_source_min: int
) -> list[dict]:
    """
    Build the candidate pool that actually reaches the cross-encoder.

    Taking a plain global top_n from the fused ranking means a source
    document whose best chunk merely scores *weaker* than another
    document's chunks — not irrelevant, just lower-ranked — can be shut
    out of reranking entirely. That's fatal for cross-document questions:
    once a source has zero candidates here, no later step (reranking,
    diversify-by-source) can bring it back, because there's nothing left
    of it to rerank or diversify with.

    So: take the global top_n as usual, then top up with each source's
    own best-ranked chunks (by fused order) until every source present
    anywhere in `fused` has contributed at least `per_source_min`
    candidates, even if that means reaching past the global cutoff.
    """
    candidates = list(fused[:top_n])
    have_ids = {c["chunk_id"] for c in candidates}

    by_source: dict[str, list[dict]] = {}
    for chunk in fused:
        by_source.setdefault(chunk["source"], []).append(chunk)

    for source, chunks in by_source.items():
        count = sum(1 for c in chunks if c["chunk_id"] in have_ids)
        for chunk in chunks:
            if count >= per_source_min:
                break
            if chunk["chunk_id"] in have_ids:
                continue
            candidates.append(chunk)
            have_ids.add(chunk["chunk_id"])
            count += 1

    return candidates


def _diversify_by_source(
    reranked: list[tuple[dict, float]], top_k: int
) -> list[tuple[dict, float]]:
    """
    Pick the final top_k reranked candidates, but guarantee the single
    best-scoring chunk from every distinct source document present in the
    candidate pool gets a slot first, before filling remaining slots by
    score alone.

    Without this, a plain top-k-by-score cut can let one document's
    chunks (e.g. many near-duplicate, overlap-split chunks all scoring
    fairly well) crowd out the only relevant chunk from a second
    document — which is exactly what breaks cross-document comparison
    questions ("what's the difference between X's policy and Y's
    policy?"), where the answer needs at least one chunk from each side.
    """
    selected: list[tuple[dict, float]] = []
    seen_sources: set[str] = set()

    for chunk, score in reranked:
        if len(selected) >= top_k:
            break
        if chunk["source"] not in seen_sources:
            selected.append((chunk, score))
            seen_sources.add(chunk["source"])

    selected_ids = {c["chunk_id"] for c, _ in selected}
    for chunk, score in reranked:
        if len(selected) >= top_k:
            break
        if chunk["chunk_id"] not in selected_ids:
            selected.append((chunk, score))
            selected_ids.add(chunk["chunk_id"])

    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[:top_k]
