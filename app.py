"""
app.py
------
Advanced RAG chatbot: multi-document upload, hybrid retrieval (vector +
BM25 + reranking), conversational query rewriting, streaming answers
with source citations.

Run locally:
    streamlit run app.py

Requires a free Groq API key — see README.md for setup.
"""

import os

import streamlit as st

from bm25_search import BM25Index
from chunker import chunk_text
from config import INDEX_DIR
from document_loader import extract_text
from hybrid_retriever import retrieve_multi
from llm_client import condense_query, decompose_query, get_client, stream_answer
from vector_store import VectorStore

st.set_page_config(page_title="Advanced RAG Chatbot", page_icon="📚", layout="wide")


def get_api_key() -> str | None:
    """Check Streamlit secrets first (for deployment), then env var (local)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("GROQ_API_KEY")


def init_session_state():
    if "vector_store" not in st.session_state:
        vs = VectorStore()
        vs.load(INDEX_DIR)  # silently no-ops if nothing saved yet
        st.session_state.vector_store = vs

    if "bm25_index" not in st.session_state:
        bm25 = BM25Index()
        if st.session_state.vector_store.chunks:
            bm25.build(st.session_state.vector_store.chunks)
        st.session_state.bm25_index = bm25

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role", "content"}


def process_uploaded_files(uploaded_files):
    vs: VectorStore = st.session_state.vector_store
    all_new_chunks = []

    progress = st.progress(0.0, text="Processing documents...")
    for i, f in enumerate(uploaded_files):
        try:
            text = extract_text(f.getvalue(), f.name)
        except ValueError as e:
            st.error(f"{f.name}: {e}")
            continue

        if not text.strip():
            st.warning(f"{f.name}: no text extracted, skipping.")
            continue

        chunks = chunk_text(text, source=f.name)
        all_new_chunks.extend(chunks)
        progress.progress((i + 1) / len(uploaded_files), text=f"Processed {f.name}")

    progress.empty()

    if all_new_chunks:
        vs.add(all_new_chunks)
        vs.save(INDEX_DIR)
        st.session_state.bm25_index.build(vs.chunks)
        st.success(f"Added {len(all_new_chunks)} chunks from {len(uploaded_files)} document(s).")


def render_sidebar():
    with st.sidebar:
        st.header("📁 Documents")

        api_key = get_api_key()
        if not api_key:
            st.error(
                "No Groq API key found. Add it to `.streamlit/secrets.toml` "
                "locally, or Streamlit Cloud's app settings. See README.md."
            )

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )
        if st.button("Process documents", disabled=not uploaded_files):
            process_uploaded_files(uploaded_files)

        vs: VectorStore = st.session_state.vector_store
        if vs.document_names:
            st.divider()
            st.caption(f"📚 {len(vs.document_names)} document(s) indexed:")
            for name in vs.document_names:
                st.caption(f"  • {name}")

            if st.button("🗑️ Clear all documents", type="secondary"):
                vs.clear(INDEX_DIR)
                st.session_state.bm25_index = BM25Index()
                st.session_state.chat_history = []
                st.rerun()
        else:
            st.info("Upload and process documents to start chatting.")

        st.divider()
        if st.button("🔄 New conversation"):
            st.session_state.chat_history = []
            st.rerun()


def render_chat():
    st.title("📚 Advanced RAG Chatbot")
    st.caption(
        "Hybrid retrieval (vector + BM25) with cross-encoder reranking, "
        "conversational query rewriting, and source citations."
    )

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("📎 Sources used"):
                    for src in turn["sources"]:
                        st.markdown(
                            f"**{src['source']}** "
                            f"(relevance: {src['rerank_score']:.2f})"
                        )
                        st.caption(src["text"][:300] + "...")

    vs: VectorStore = st.session_state.vector_store
    api_key = get_api_key()

    question = st.chat_input(
        "Ask a question about your documents...",
        disabled=not (vs.chunks and api_key),
    )

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        client = get_client(api_key)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                standalone_query = condense_query(
                    client, question, st.session_state.chat_history[:-1]
                )
                sub_queries = decompose_query(client, standalone_query)
                chunks = retrieve_multi(
                    sub_queries, vs, st.session_state.bm25_index
                )

            if not chunks:
                answer = (
                    "I couldn't find anything relevant to that question in "
                    "the uploaded documents."
                )
                st.markdown(answer)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "sources": []}
                )
            else:
                answer_stream = stream_answer(client, standalone_query, chunks)
                full_answer = st.write_stream(answer_stream)

                with st.expander("📎 Sources used"):
                    for src in chunks:
                        st.markdown(
                            f"**{src['source']}** "
                            f"(relevance: {src['rerank_score']:.2f})"
                        )
                        st.caption(src["text"][:300] + "...")

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_answer, "sources": chunks}
                )


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
