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
# App & Global Setup
# ======================================================

app = FastAPI(
    title="Santhi RAG Backend",
    version="1.0.0",
    description="FastAPI + Groq + FastEmbed RAG backend with chunking & ingestion.",
)

# CORS for local dev + deployed frontend (you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # e.g. ["http://localhost:5173", "https://santhi-rag-frontend.vercel.app"]
    allow_credentials=False,  # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create DB tables at startup
create_db()

# Embedding model (global singleton to avoid re-loading)
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Groq client + model name (override via env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

# Default is a valid Groq model; override via GROQ_MODEL_NAME if needed
GROQ_MODEL = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")

client = Groq(api_key=GROQ_API_KEY)


# ======================================================
# Helper: Groq call wrapper
# ======================================================

def call_groq_chat(messages: List[dict]) -> str:
    """
    Wrapper around Groq chat completion.
    Converts all errors to HTTP 500 with a clear message.
    """
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # Don't leak internal stack traces to the client
        raise HTTPException(
            status_code=500,
            detail=f"Groq API error: {str(e)}",
        )


# ======================================================
# Helper: Chunking
# ======================================================

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    max_chunks: int = 512,  # safety cap to avoid OOM
) -> List[str]:
    """
    Word-level sliding window chunking.
    - chunk_size : number of words per chunk
    - overlap    : words to overlap between consecutive chunks
    - max_chunks : hard limit on chunks
    """
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words) and len(chunks) < max_chunks:
        end = start + chunk_size
        chunk = words[start:end]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
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
    return {
        "status": "OK",
        "message": "RAG backend online",
        "model": GROQ_MODEL,
    }


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
    - Stores question, answer, and embedding in ChatLog
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    answer = call_groq_chat(
        messages=[{"role": "user", "content": question}]
    )

    # Embed question using FastEmbed; ensure pure Python float list
    raw_vec = list(embedder.embed([question]))[0]
    q_vec = [float(x) for x in raw_vec]

    log = ChatLog(
        question=question,
        answer=answer,
        embedding=q_vec,
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
    raw_text = (payload.text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="No text to ingest.")

    # 1) Document metadata
    doc = IngestedDocument(
        title=payload.title.strip() or "Untitled",
        source="text",
        original_path=None,
        collection=payload.collection or "default",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # 2) Chunk
    chunks = chunk_text(raw_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Chunking produced no chunks.")

    # 3) Embed chunks
    embeddings = list(embedder.embed(chunks))

    # 4) Store chunks (convert to pure float)
    for idx, (chunk_text_str, emb) in enumerate(zip(chunks, embeddings)):
        vec = [float(x) for x in emb]
        chunk_row = IngestedChunk(
            document_id=doc.id,
            chunk_index=idx,
            text=chunk_text_str,
            embedding=vec,
            collection=payload.collection or "default",
        )
        session.add(chunk_row)

    session.commit()

    return {
        "status": "ok",
        "document_id": doc.id,
        "chunks": len(chunks),
        "collection": payload.collection or "default",
    }


# ======================================================
# 3) PDF INGESTION
# ======================================================

MAX_PDF_BYTES = 5_000_000  # 5 MB


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

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF uploaded.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF too large (max 5 MB).")

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")

    if not full_text:
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")

    # 1) Document metadata
    doc = IngestedDocument(
        title=(title or file.filename or "Untitled").strip(),
        source="pdf",
        original_path=file.filename,
        collection=collection or "default",
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
        vec = [float(x) for x in emb]
        session.add(
            IngestedChunk(
                document_id=doc.id,
                chunk_index=idx,
                text=chunk_text_str,
                embedding=vec,
                collection=collection or "default",
            )
        )

    session.commit()

    return {
        "status": "ok",
        "document_id": doc.id,
        "chunks": len(chunks),
        "collection": collection or "default",
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
    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1) Embed query
    q_vec = np.array(list(embedder.embed([question]))[0], dtype="float32")

    # 2) Fetch candidate chunks for this collection
    rows: List[IngestedChunk] = session.exec(
        select(IngestedChunk).where(IngestedChunk.collection == (collection or "default"))
    ).all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No chunks found for collection '{collection}'. Ingest docs first.",
        )

    # 3) Compute cosine similarity
    scored: List[Tuple[float, IngestedChunk]] = []
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
    top = scored[: max(1, top_k)]

    # 4) Build context
    context = "\n\n".join(
        [f"[Chunk {r.chunk_index} | score={s:.4f}]\n{r.text}" for (s, r) in top]
    )

    prompt = f"""
You are an AI assistant. Use ONLY the context below to answer the user question.
If the answer is not present in the context, say explicitly that you don't know.

--- CONTEXT START ---
{context}
--- CONTEXT END ---

Question: {question}

Answer clearly and concisely based on the context.
"""

    # 5) Call Groq LLM
    final_answer = call_groq_chat(
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "query": question,
        "collection": collection or "default",
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

    question = (q or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def token_stream():
        try:
            stream = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": question}],
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
                    # tiny delay so clients see streaming effect
                    await asyncio.sleep(0.01)

            yield "\n[END]"
        except Exception as e:
            # On streaming error, send one error line then end
            yield f"\n[STREAM ERROR: {str(e)}]\n"

    return StreamingResponse(token_stream(), media_type="text/plain")
