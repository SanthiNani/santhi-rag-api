# app/main.py

import os
import io
import asyncio
from typing import List, Optional

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
# App & Global Setup
# ======================================================

app = FastAPI(
    title="Santhi RAG Backend",
    version="1.0.0",
    description="FastAPI + Groq + FastEmbed RAG backend with chunking & ingestion.",
)

# Basic CORS so frontend (Render / Vercel / localhost) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables at startup
create_db()

# Embedding model (lightweight, CPU-friendly, free)
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Fail fast at container startup if key is missing
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)


# ======================================================
# Helper: Chunking
# ======================================================

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[str]:
    """
    Word-level sliding window chunking.
    - chunk_size : number of words per chunk
    - overlap    : words to overlap between consecutive chunks
    """
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        # Slide window with overlap
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
# Health / Root
# ======================================================

@app.get("/")
def root():
    return {"status": "OK", "message": "RAG backend online"}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ======================================================
# 1) Normal ASK (chat + store in ChatLog with embeddings)
# ======================================================

@app.post("/ask")
def ask(
    payload: AskRequest,
    session: Session = Depends(get_session),
):
    """
    Simple Q&A:
    - Sends question to Groq LLM
    - Stores question, answer, and embedding in ChatLog (for future semantic search)
    """
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": payload.question}],
    )

    answer = completion.choices[0].message.content

    # Embed question using FastEmbed
    q_vec = list(embedder.embed([payload.question]))[0]

    log = ChatLog(
        question=payload.question,
        answer=answer,
        embedding=list(q_vec),
    )
    session.add(log)
    session.commit()

    return {"answer": answer}


# ======================================================
# 2) TEXT INGESTION (chunks + embeddings)
# ======================================================

@app.post("/ingest_text")
async def ingest_text(
    payload: IngestTextRequest,
    session: Session = Depends(get_session),
):
    """
    Ingest raw text into the RAG store:
    - Create IngestedDocument
    - Chunk text
    - Embed each chunk
    - Store in IngestedChunk
    """
    # 1) Document metadata
    doc = IngestedDocument(
        title=payload.title,
        source="text",
        original_path=None,
        collection=payload.collection,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # 2) Chunk
    chunks = chunk_text(payload.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text to ingest.")

    # 3) Embed chunks
    embeddings = list(embedder.embed(chunks))

    # 4) Store chunks
    for idx, (chunk_text_str, emb) in enumerate(zip(chunks, embeddings)):
        chunk_row = IngestedChunk(
            document_id=doc.id,
            chunk_index=idx,
            text=chunk_text_str,
            embedding=list(emb),
            collection=payload.collection,
        )
        session.add(chunk_row)

    session.commit()

    return {
        "status": "ok",
        "document_id": doc.id,
        "chunks": len(chunks),
        "collection": payload.collection,
    }


# ======================================================
# 3) PDF INGESTION
# ======================================================

@app.post("/ingest_pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    collection: str = Form("default"),
    session: Session = Depends(get_session),
):
    """
    Ingest a PDF:
    - Read uploaded PDF bytes
    - Extract text from all pages
    - Chunk + embed + store
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")

    if not full_text:
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")

    title = title or file.filename

    # 1) Document metadata
    doc = IngestedDocument(
        title=title,
        source="pdf",
        original_path=file.filename,
        collection=collection,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # 2) Chunk text
    chunks = chunk_text(full_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Chunking produced no chunks.")

    # 3) Embed chunks
    embeddings = list(embedder.embed(chunks))

    # 4) Store in DB
    for idx, (chunk_text_str, emb) in enumerate(zip(chunks, embeddings)):
        session.add(
            IngestedChunk(
                document_id=doc.id,
                chunk_index=idx,
                text=chunk_text_str,
                embedding=list(emb),
                collection=collection,
            )
        )

    session.commit()

    return {
        "status": "ok",
        "document_id": doc.id,
        "chunks": len(chunks),
        "collection": collection,
    }


# ======================================================
# 4) RAG ANSWER (retrieval + generation)
# ======================================================

@app.get("/rag_answer")
def rag_answer(
    q: str,
    top_k: int = 5,
    collection: str = "default",
    session: Session = Depends(get_session),
):
    """
    Full RAG pipeline:
    - Embed query
    - Retrieve top-k similar chunks from IngestedChunk
    - Build context string
    - Ask Groq LLM to answer using ONLY that context
    """
    # 1) Embed query
    q_vec = np.array(list(embedder.embed([q]))[0], dtype="float32")

    # 2) Fetch candidate chunks for this collection
    rows: List[IngestedChunk] = session.exec(
        select(IngestedChunk).where(IngestedChunk.collection == collection)
    ).all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for collection '{collection}'. Ingest docs first.",
        )

    # 3) Compute cosine similarity
    scored: List[tuple[float, IngestedChunk]] = []
    q_norm = float(np.linalg.norm(q_vec))
    if q_norm == 0.0:
        raise HTTPException(status_code=400, detail="Query embedding norm is zero.")

    for r in rows:
        emb = np.array(r.embedding, dtype="float32")
        emb_norm = float(np.linalg.norm(emb))
        if emb_norm == 0.0:
            continue
        sim = float(np.dot(q_vec, emb) / (q_norm * emb_norm))
        scored.append((sim, r))

    if not scored:
        raise HTTPException(
            status_code=404,
            detail="No valid embeddings found in chunks.",
        )

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    # 4) Build context
    context = "\n\n".join(
        [f"[Chunk {r.chunk_index} | score={s:.4f}]\n{r.text}" for (s, r) in top]
    )

    prompt = f"""
You are an AI assistant. Use ONLY the context below to answer the user question.
If the answer is not present in the context, say you don't know.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {q}

Answer clearly and concisely based on the context.
"""

    # 5) Call Groq LLM
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )

    final_answer = resp.choices[0].message.content

    return {
        "query": q,
        "collection": collection,
        "retrieved_chunks": [
            {
                "score": float(s),
                "chunk_id": r.id,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "text_preview": r.text[:200],
            }
            for (s, r) in top
        ],
        "final_answer": final_answer,
    }


# ======================================================
# 5) Streaming endpoint (no RAG, direct LLM chat)
# ======================================================

@app.get("/stream")
async def stream_answer(q: str):
    """
    Streams answer token-by-token from Groq, ChatGPT-style.
    """

    async def token_stream():
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": q}],
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
                # tiny delay so clients see streaming effect
                await asyncio.sleep(0.01)

        yield "\n[END]"

    return StreamingResponse(token_stream(), media_type="text/plain")
