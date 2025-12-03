from fastapi import FastAPI
from app.db import ChatLog, create_db, get_session
from typing import Optional, List
from sqlmodel import select
from pydantic import BaseModel
from groq import Groq
import os
import numpy as np
from fastapi.responses import StreamingResponse
import asyncio

from sentence_transformers import SentenceTransformer

app = FastAPI()

create_db()

# FREE embeddings model
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class AskRequest(BaseModel):
    question: str

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.get("/")
def root():
    return {"status": "OK", "message": "LLM API Online"}


# -----------------------------------------------------------
# 1) NORMAL ASK ENDPOINT
# -----------------------------------------------------------
@app.post("/ask")
def ask(payload: AskRequest):
    # generate answer using Groq Llama
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": payload.question}]
    )
    answer = completion.choices[0].message.content

    # FREE embeddings for semantic search
    emb = embedder.encode(payload.question).tolist()

    with get_session() as session:
        log = ChatLog(question=payload.question, answer=answer, embedding=emb)
        session.add(log)
        session.commit()

    return {"answer": answer}


# -----------------------------------------------------------
# 2) SEMANTIC SEARCH ENDPOINT
# -----------------------------------------------------------
@app.get("/semantic_search")
def semantic_search(q: str, limit: int = 5):
    q_emb = embedder.encode(q)

    with get_session() as session:
        rows = session.exec(select(ChatLog)).all()

    scored = []
    for r in rows:
        if r.embedding:
            db_emb = np.array(r.embedding)
            num = np.dot(q_emb, db_emb)
            den = (np.linalg.norm(q_emb) * np.linalg.norm(db_emb))
            cos = float(num / den)
            scored.append((cos, r))

    # highest similarity first
    scored.sort(reverse=True, key=lambda x: x[0])

    return [
        {
            "score": s,
            "question": r.question,
            "answer": r.answer,
            "created_at": r.created_at.isoformat()
        }
        for (s, r) in scored[:limit]
    ]


# -----------------------------------------------------------
# 3) FULL RAG PIPELINE
# -----------------------------------------------------------
@app.get("/rag_answer")
def rag_answer(q: str, top_k: int = 3):
    """
    1) Embed the query
    2) Run semantic search
    3) Build RAG context
    4) Ask Groq Llama with context
    """

    # 1) embed query
    q_emb = embedder.encode(q)

    # load all logs
    with get_session() as session:
        rows = session.exec(select(ChatLog)).all()

    scored = []
    for r in rows:
        if r.embedding:
            db_emb = np.array(r.embedding)
            cos = float(np.dot(q_emb, db_emb) /
                        (np.linalg.norm(q_emb) * np.linalg.norm(db_emb)))
            scored.append((cos, r))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_docs = scored[:top_k]

    # 2) build context
    context = "\n\n".join(
        [f"Q: {r.question}\nA: {r.answer}" for (_, r) in top_docs]
    )

    # 3) final RAG prompt
    prompt = f"""
You are an AI assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question:
{q}

Answer clearly and concisely based on the context.
"""

    # 4) call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    final_answer = response.choices[0].message.content

    return {
        "query": q,
        "retrieved_docs": [
            {
                "score": s,
                "question": r.question,
                "answer": r.answer
            }
            for (s, r) in top_docs
        ],
        "final_answer": final_answer
    }

# -----------------------------------------------------------
# 4) STREAMING ENDPOINT (DAY 8)
# -----------------------------------------------------------
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/stream")
async def stream_answer(q: str):
    """
    Streams answer token-by-token like ChatGPT
    """

    async def token_generator():
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": q}],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                yield token
                await asyncio.sleep(0.01)

        yield "\n[END]"

    return StreamingResponse(token_generator(), media_type="text/plain")
