# 📚 Advanced RAG Chatbot

A production-style Retrieval-Augmented Generation chatbot — not a basic
"embed and cosine-similarity-search" tutorial. Upload multiple documents,
ask questions across all of them in a real conversation, and get answers
grounded in your documents with source citations.

**Live demo:**  [RAG Chatbot App](https://rag-chatbot-shivampanday.streamlit.app/)

---

## 🧠 What makes this "advanced"

| Feature | Why it matters |
|---|---|
| **Hybrid retrieval** (vector + BM25) | Vector search alone misses exact keyword/code/name matches; BM25 alone misses paraphrases. Combining both, via Reciprocal Rank Fusion, catches more relevant chunks than either alone. |
| **Cross-encoder reranking** | A second, more precise model re-scores the top candidates by looking at the query and chunk *together* — significantly more accurate than comparing independently-computed embeddings, which is all basic RAG does. |
| **Conversational query rewriting** | Follow-up questions like "what about the second one?" get rewritten into standalone queries using chat history *before* retrieval — without this, multi-turn conversations break. |
| **Multi-document support + citations** | Upload several files, ask cross-document questions, see exactly which document backed each part of the answer. |
| **Persistent vector index** | Saved to disk — documents aren't re-embedded every time you restart the app. |
| **Streaming answers** | Tokens appear as they're generated, not a blocking wait. |

## 🏗️ Architecture

```
User question
    │
    ▼
Query condensation (LLM rewrites follow-ups → standalone query)
    │
    ▼
┌─────────────────┐       ┌─────────────────┐
│  Vector search   │       │   BM25 search    │
│  (FAISS, dense)  │       │ (keyword, sparse)│
└────────┬─────────┘       └────────┬─────────┘
         │                          │
         └──────────┬───────────────┘
                     ▼
         Reciprocal Rank Fusion (merge rankings)
                     │
                     ▼
         Cross-encoder reranking (precise re-score)
                     │
                     ▼
         Top-K chunks → LLM (streamed, grounded answer)
```

## 📁 Project Structure

```
rag-chatbot/
├── app.py                 ← Streamlit UI (chat, uploads, citations)
├── document_loader.py     ← extracts text from PDF/DOCX/TXT
├── chunker.py              ← recursive chunking with overlap
├── vector_store.py         ← FAISS wrapper (build/search/save/load)
├── bm25_search.py           ← BM25 keyword index
├── hybrid_retriever.py      ← RRF fusion + cross-encoder reranking
├── llm_client.py             ← Groq API: query rewriting + streaming answers
├── config.py                 ← all tunable constants in one place
├── requirements.txt
├── runtime.txt                ← pins Python 3.11 for Streamlit Cloud
├── .streamlit/secrets.toml.example
├── .gitignore
└── README.md
```

## 🔑 Getting a free Groq API key

This project uses [Groq](https://console.groq.com) for the LLM — it's free-tier
friendly and very fast. No credit card needed for the free tier.

1. Go to [console.groq.com](https://console.groq.com) and sign up
2. Go to **API Keys** → **Create API Key**
3. Copy the key (you won't be able to see it again)

## ⚙️ Setup

1. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # Mac/Linux
   python -m pip install -r requirements.txt
   ```

2. **Add your API key locally:** 
   Create a new folder '.streamlit':
   ```bash
   mkdir .streamlit
   ```
   Create a new file 'secrets.toml' in '.streamlit' folder, then open the 'secrets.toml' file and type this exactly one line shown below and paste your real Groq API key in there:
   ```toml
   GROQ_API_KEY = "your-actual-key-here"
   ```
   This file is gitignored — it will never be pushed to GitHub.

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`.

4. **Use it:**
   - Upload one or more PDF/DOCX/TXT files in the sidebar
   - Click "Process documents" (embeds and indexes them — takes a few seconds)
   - Ask questions in the chat box
   - Expand "Sources used" under any answer to see which document/chunk it came from

## 🚀 Deploying for free (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, click **New app**, select this repo, set main file to `app.py`.
3. **Before deploying**, click **Advanced settings** and set the Python
   version to **3.11** — this avoids a known issue where newer Python
   versions break some of this project's dependencies (see `runtime.txt`,
   which also pins this, as a second safety net).
4. **Add your secret:** after deploying (or in Advanced settings before
   deploying), go to your app's **Settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your-actual-key-here"
   ```
   This is the cloud equivalent of your local `.streamlit/secrets.toml`.
5. Deploy. First load will take a minute or two while it downloads the
   embedding and reranker models.

## ✍️ Resume bullet (once deployed)

> Built an advanced RAG chatbot with hybrid retrieval (FAISS vector search
> plus BM25, fused via Reciprocal Rank Fusion) and cross-encoder reranking,
> supporting multi-document Q&A with conversational query rewriting,
> streaming answers, and source citations.

