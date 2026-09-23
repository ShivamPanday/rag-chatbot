"""
llm_client.py
--------------
Wraps the Groq API (free-tier friendly, fast inference) for three jobs:
  1. Query condensation: rewrite a follow-up question into a standalone
     query using chat history, so retrieval works correctly on things
     like "what about the second one?"
  2. Query decomposition: split a compound/comparison question ("what's
     the difference between A's X and B's Y") into separate, self-
     contained sub-questions, so retrieval for one topic isn't diluted
     or out-competed by terms from the other.
  3. Answer generation: stream a grounded answer from the retrieved
     context, citing which sources it used.
"""

import json

from groq import Groq

from config import LLM_MODEL

CONDENSE_SYSTEM_PROMPT = (
    "Given a chat history and a follow-up question, rewrite the follow-up "
    "into a standalone question that contains all necessary context from "
    "the history. If the follow-up question is already standalone, return "
    "it unchanged. Output ONLY the rewritten question, nothing else."
)

DECOMPOSE_SYSTEM_PROMPT = (
    "You turn a user's question into one or more self-contained search "
    "queries for a document retrieval system.\n"
    "- If the question compares, contrasts, or otherwise asks about two "
    "or more distinct topics, entities, or documents (e.g. \"What's the "
    "difference between A's policy and B's policy?\"), split it into one "
    "separate, self-contained question per topic — each phrased so it "
    "could be searched and answered entirely on its own, without needing "
    "the other topic for context.\n"
    "- If the question is about a single topic, return it unchanged as "
    "the only item.\n"
    "Respond with ONLY a JSON array of strings, nothing else — no "
    "markdown, no explanation. Example: [\"question one\", \"question two\"]"
)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions based ONLY on the "
    "provided context excerpts. Each excerpt is labeled with its source "
    "document. Follow these rules strictly:\n"
    "1. Only use information from the provided context — do not use "
    "outside knowledge.\n"
    "2. If the context doesn't contain enough information to answer, say "
    "so clearly rather than guessing.\n"
    "3. When you use information from an excerpt, mention which source "
    "document it came from, e.g. '(from report.pdf)'.\n"
    "4. Be concise and direct."
)


def get_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def condense_query(client: Groq, question: str, chat_history: list[dict]) -> str:
    """Rewrite a follow-up question into a standalone query using history.
    Skips the LLM call entirely if there's no history yet (first turn)."""
    if not chat_history:
        return question

    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in chat_history[-6:]
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Chat history:\n{history_text}\n\nFollow-up question: {question}",
            },
        ],
        temperature=0,
        max_tokens=300,
    )
    condensed = response.choices[0].message.content.strip()

    # Safety net: if the rewrite came back empty, got cut off mid-sentence
    # (hit the token cap), or is suspiciously short compared to the
    # original, fall back to the original question rather than risk
    # sending a mangled/truncated query downstream.
    finish_reason = response.choices[0].finish_reason
    looks_truncated = finish_reason == "length" or (
        condensed and condensed[-1] not in ".?!\"'"
    )
    if not condensed or looks_truncated:
        return question

    return condensed


def decompose_query(client: Groq, question: str) -> list[str]:
    """
    Split a compound/comparison question into independent sub-questions,
    one per topic. Retrieving each sub-question separately (see
    hybrid_retriever.retrieve_multi) stops one topic's terms from
    diluting or out-ranking the other's relevant chunk — which is what
    happens when a single blended query like "what's the difference
    between A's policy and B's policy?" is retrieved as-is.

    Falls back to [question] unchanged on any parsing failure, so a
    decomposition hiccup degrades to the old single-query behavior
    rather than breaking retrieval entirely.
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        if (
            isinstance(parsed, list)
            and parsed
            and all(isinstance(q, str) and q.strip() for q in parsed)
        ):
            return [q.strip() for q in parsed[:4]]  # cap sub-queries for latency/cost
    except Exception:
        pass
    return [question]


def stream_answer(client: Groq, question: str, chunks: list[dict]):
    """
    Generate a streaming answer grounded in the retrieved chunks.
    Yields text fragments as they arrive from the API.
    """
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Source: {c['source']}]\n{c['text']}")
    context_text = "\n\n---\n\n".join(context_blocks)

    stream = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context excerpts:\n\n{context_text}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
