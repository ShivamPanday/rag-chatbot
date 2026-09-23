"""
config.py
----------
Central configuration for the RAG pipeline. Tweak these to trade off
speed vs. quality without touching the rest of the codebase.
"""

# Chunking
CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 150       # overlap between consecutive chunks

# Retrieval
VECTOR_TOP_K = 12         # candidates pulled from FAISS
BM25_TOP_K = 12           # candidates pulled from BM25
RRF_K = 60                # Reciprocal Rank Fusion constant (standard default)
RERANK_TOP_N = 10         # how many fused candidates to rerank (was 5 — too
                           # tight; relevant chunks were getting cut before
                           # the cross-encoder ever scored them, especially
                           # for cross-document comparison questions)
PER_SOURCE_MIN_CANDIDATES = 3  # guarantee each source document contributes
                           # at least this many of its own best-fused
                           # chunks to the reranking pool, even if none of
                           # them would otherwise survive the global
                           # RERANK_TOP_N cut. Without this, a document
                           # that scores weaker overall on a given query
                           # (e.g. its best chunk is about "escalation
                           # path" while the query also mentions a whole
                           # other product's "data export") can be shut
                           # out of reranking entirely — and no amount of
                           # diversification after reranking can recover
                           # a source that never got there.
FINAL_TOP_K = 6           # how many reranked chunks go to the LLM (was 4 —
                           # too small to reliably cover both a generic and
                           # a more specific chunk on the same topic, e.g.
                           # an overlap-split "standard vs. enterprise" pair)
MERGED_TOP_K = 10          # cap on chunks kept after merging results from
                           # decomposed sub-queries (see retrieve_multi in
                           # hybrid_retriever.py)

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "openai/gpt-oss-120b"  # Groq-hosted, fast + free-tier friendly

# Storage
INDEX_DIR = "data/index"
