# app/main.py

import os
import io
import asyncio
from typing import List, Optional, Tuple

import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from groq import Groq
from fastembed import TextEmbedding
from pypdf import PdfReader

from app.db import (
    ChatLog,
    IngestedDocument,
    IngestedChunk,
    create_db,
    get_session,
)

# ======================================================
# App Setup
# ======================================================

app = FastAPI(
    title="Santhi RAG Backend",
    version="1.0.1",
    description="FastAPI + Groq + FastEmbed RAG backend with chunking & ingestion.",
)

# CORS (wide open for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB init
create_db()

# Embedding Model
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Groq Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment variables.")

# Default correct model name
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

client = Groq(api_key=GROQ_API_KEY)


# ======================================================
# Utility: Convert embeddings to Python floats
# ======================================================

def to_float_list(arr) -> List[float]:
    """Convert numpy/fastembed array to JSON-safe float list."""
    return [float(x) for x in arr]


# ======================================================
# Utility: Groq wrapper
# ======================================================

def call_groq_chat(messages: List[dict]) -> str:
    """
    Wrapper around Groq chat completion.
    Ensures model errors + empty responses are handled safely.
    """
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
        )

        content = completion.choices[0].message.content

        # 🔥 FIX: Groq sometimes returns None → force safe non-empty answer
        if not content or not isinstance(content, str) or content.strip() == "":
            return "(Model returned no answer.)"

        return content.strip()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Groq API error: {str(e)}",
        )


# ======================================================
# Helper: Chunking
# ======================================================

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ======================================================
# Schemas
# ======================================================

class AskRequest(BaseModel):
    question: str


class IngestTextRequest(BaseModel):
    title: str
    text: str
    collection: Optional[str] = "default"


# ======================================================
# Health
# ======================================================

@app.get("/")
def root():
    return {
        "status": "OK",
        "model": GROQ_MODEL,
        "message": "RAG Backend Running",
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ======================================================
# ASK Endpoint
# ======================================================

@app.post("/ask")
def ask(payload: AskRequest, session: Session = Depends(get_session)):

    answer = call_groq_chat(messages=[{"role": "user", "content": payload.question}])

    # Embed question
    vec_raw = list(embedder.embed([payload.question]))[0]
    vec = to_float_list(vec_raw)

    log = ChatLog(
        question=payload.question,
        answer=answer,
        embedding=vec,
    )
    session.add(log)
    session.commit()

    return {"answer": answer}


# ======================================================
# TEXT INGESTION
# ======================================================

@app.post("/ingest_text")
async def ingest_text(payload: IngestTextRequest, session: Session = Depends(get_session)):

    doc = IngestedDocument(
        title=payload.title,
        source="text",
        original_path=None,
        collection=payload.collection,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    chunks = chunk_text(payload.text)
    if not chunks:
        raise HTTPException(400, "No content to ingest.")

    embeddings = list(embedder.embed(chunks))

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        session.add(
            IngestedChunk(
                document_id=doc.id,
                chunk_index=i,
                text=chunk,
                embedding=to_float_list(emb),
                collection=payload.collection,
            )
        )

    session.commit()

    return {"status": "ok", "chunks": len(chunks), "document_id": doc.id}


# ======================================================
# PDF INGESTION
# ======================================================

MAX_PDF_BYTES = 5_000_000  # 5MB


@app.post("/ingest_pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    collection: str = Form("default"),
    session: Session = Depends(get_session),
):

    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF allowed.")

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(400, "Empty PDF.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(413, "PDF too large (>5MB).")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join([page.extract_text() or "" for page in reader.pages]).strip()
    except Exception as e:
        raise HTTPException(500, f"PDF parsing error: {e}")

    if not text:
        raise HTTPException(400, "No text extracted from PDF.")

    title = title or file.filename

    doc = IngestedDocument(
        title=title,
        source="pdf",
        original_path=title,
        collection=collection,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    chunks = chunk_text(text)
    embeddings = list(embedder.embed(chunks))

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        session.add(
            IngestedChunk(
                document_id=doc.id,
                chunk_index=i,
                text=chunk,
                embedding=to_float_list(emb),
                collection=collection,
            )
        )

    session.commit()
    return {"status": "ok", "chunks": len(chunks), "document_id": doc.id}


# ======================================================
# RAG ANSWER
# ======================================================

@app.get("/rag_answer")
def rag_answer(
    q: str,
    top_k: int = 5,
    collection: str = "default",
    session: Session = Depends(get_session),
):

    q_vec_raw = list(embedder.embed([q]))[0]
    q_vec = np.array(to_float_list(q_vec_raw), dtype="float32")

    rows = session.exec(
        select(IngestedChunk).where(IngestedChunk.collection == collection)
    ).all()

    if not rows:
        raise HTTPException(404, "No chunks found. Ingest documents first.")

    scored = []
    q_norm = float(np.linalg.norm(q_vec))

    for r in rows:
        emb = np.array(r.embedding, dtype="float32")
        emb_norm = float(np.linalg.norm(emb))
        if emb_norm == 0:
            continue
        sim = float(np.dot(q_vec, emb) / (q_norm * emb_norm))
        scored.append((sim, r))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    context = "\n\n".join(
        [f"[Chunk {r.chunk_index} | score={s:.4f}]\n{r.text}" for (s, r) in top]
    )

    prompt = f"""
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know".

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {q}
"""

    final_answer = call_groq_chat(
        messages=[{"role": "user", "content": prompt}]
    )
    
    if not final_answer or final_answer.strip() == "":
        final_answer = "(RAG model returned no answer.)"

    return {
        "answer": final_answer,
        "retrieved": [
            {"chunk_id": r.id, "score": float(s), "text_preview": r.text[:200]}
            for (s, r) in top
        ],
    }


# ======================================================
# STREAMING
# ======================================================

@app.get("/stream")
async def stream_answer(q: str):

    async def token_stream():
        try:
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": q}],
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
                    await asyncio.sleep(0.008)

            yield "\n[END]"
        except Exception as e:
            yield f"\n[STREAM ERROR: {e}]\n"

    return StreamingResponse(token_stream(), media_type="text/plain")
