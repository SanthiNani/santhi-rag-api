# app/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List

import os
import asyncio
import numpy as np

from dotenv import load_dotenv
from groq import Groq
from fastembed import TextEmbedding

from app.db import ChatLog, create_db, get_session


# ---------------------------------------------------------
# INIT
# ---------------------------------------------------------
load_dotenv()

app = FastAPI(title="RAG API (Groq + fastembed)")

create_db()

# lightweight embedding model (NO torch, NO CUDA)
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str) -> np.ndarray:
    """
    Compute a dense vector for input text using fastembed.
    Returns a numpy array for easy cosine similarity.
    """
    # embed returns a generator -> first item
    vec = next(embedder.embed([text]))
    return np.array(vec, dtype=np.float32)


class AskRequest(BaseModel):
    question: str


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # be explicit; otherwise you'll just get 401s and waste time
    raise RuntimeError("GROQ_API_KEY not set in environment/.env")

client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "OK", "message": "LLM API Online (Groq + fastembed)"}


# ---------------------------------------------------------
# 1) NORMAL ASK ENDPOINT
# ---------------------------------------------------------
@app.post("/ask")
def ask(payload: AskRequest):
    """
    1) Call Groq LLM with user's question.
    2) Compute embedding with fastembed.
    3) Store question, answer, embedding into SQLite.
    """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": payload.question}],
    )
    answer = completion.choices[0].message.content

    # embedding with fastembed (cheap, lightweight)
    emb = get_embedding(payload.question).tolist()

    with get_session() as session:
        log = ChatLog(question=payload.question, answer=answer, embedding=emb)
        session.add(log)
        session.commit()

    return {"answer": answer}


# ---------------------------------------------------------
# 2) SEMANTIC SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/semantic_search")
def semantic_search(q: str, limit: int = 5):
    """
    Semantic search over stored ChatLog questions based on embeddings.
    """
    q_emb = get_embedding(q)

    with get_session() as session:
        rows: List[ChatLog] = session.query(ChatLog).all()

    scored = []
    for r in rows:
        if not r.embedding:
            continue
        db_emb = np.array(r.embedding, dtype=np.float32)

        num = float(np.dot(q_emb, db_emb))
        den = float(np.linalg.norm(q_emb) * np.linalg.norm(db_emb))
        if den == 0:
            continue
        cos = num / den
        scored.append((cos, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "score": float(s),
            "question": r.question,
            "answer": r.answer,
            "created_at": r.created_at.isoformat(),
        }
        for (s, r) in scored[:limit]
    ]


# ---------------------------------------------------------
# 3) FULL RAG PIPELINE
# ---------------------------------------------------------
@app.get("/rag_answer")
def rag_answer(q: str, top_k: int = 3):
    """
    1) Embed query with fastembed.
    2) Find top-k similar past Q&A from ChatLog via cosine similarity.
    3) Build RAG context from retrieved logs.
    4) Ask Groq LLM using that context.
    """

    q_emb = get_embedding(q)

    with get_session() as session:
        rows: List[ChatLog] = session.query(ChatLog).all()

    scored = []
    for r in rows:
        if not r.embedding:
            continue
        db_emb = np.array(r.embedding, dtype=np.float32)

        num = float(np.dot(q_emb, db_emb))
        den = float(np.linalg.norm(q_emb) * np.linalg.norm(db_emb))
        if den == 0:
            continue
        cos = num / den
        scored.append((cos, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = scored[:top_k]

    context = "\n\n".join(
        [f"Q: {r.question}\nA: {r.answer}" for (_, r) in top_docs]
    )

    prompt = f"""
You are an AI assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{q}

Answer clearly and concisely based on the context.
""".strip()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    final_answer = response.choices[0].message.content

    return {
        "query": q,
        "retrieved_docs": [
            {
                "score": float(s),
                "question": r.question,
                "answer": r.answer,
            }
            for (s, r) in top_docs
        ],
        "final_answer": final_answer,
    }


# ---------------------------------------------------------
# 4) STREAMING ENDPOINT
# ---------------------------------------------------------
@app.get("/stream")
async def stream_answer(q: str):
    """
    Streams tokens from Groq like ChatGPT.
    """

    async def token_generator():
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": q}],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
                # avoid hammering; also makes Render logs readable
                await asyncio.sleep(0.01)

        yield "\n[END]"

    return StreamingResponse(token_generator(), media_type="text/plain")
