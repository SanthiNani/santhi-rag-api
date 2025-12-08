from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
from sqlmodel import select, Session
from pydantic import BaseModel
import numpy as np
import os
import asyncio

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

# -----------------------------------------
# FastAPI App Init
# -----------------------------------------
app = FastAPI()
create_db()

# -----------------------------------------
# Embedding model
# -----------------------------------------
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# -----------------------------------------
# Groq LLM Client
# -----------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------------------
# Root endpoint
# -----------------------------------------
@app.get("/")
def root():
    return {"status": "OK", "message": "RAG System Online"}


# ======================================================
# CHUNKING FUNCTION
# ======================================================

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50
) -> List[str]:
    """
    Simple whitespace-based chunking.
    - chunk_size = number of words per chunk
    - overlap = repeated words between chunks (to avoid boundary loss)
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        if not chunk:
            break

        chunks.append(" ".join(chunk))

        start = end - overlap  # sliding window with overlap

    return chunks


# ======================================================
# 1) Normal ASK (ChatLog only)
# ======================================================

class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(payload: AskRequest, session: Session = Depends(get_session)):
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": payload.question}]
    )

    answer = completion.choices[0].message.content

    # embed question using fastembed
    emb = list(embedder.embed([payload.question]))[0]

    log = ChatLog(
        question=payload.question,
        answer=answer,
        embedding=emb
    )

    session.add(log)
    session.commit()

    return {"answer": answer}


# ======================================================
# 2) TEXT INGESTION (Chunks + Embeddings)
# ======================================================

class IngestTextRequest(BaseModel):
    title: str
    text: str
    collection: Optional[str] = "default"


@app.post("/ingest_text")
async def ingest_text(
    payload: IngestTextRequest,
    session: Session = Depends(get_session)
):
    # Create document row
    doc = IngestedDocument(
        title=payload.title,
        source="text",
        original_path=None,
        collection=payload.collection,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Chunk the text
    chunks = chunk_text(payload.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted.")

    # Embed chunks
    embeddings = list(embedder.embed(chunks))

    # Store chunks
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
    session: Session = Depends(get_session)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed.")

    pdf_bytes = await file.read()

    try:
        reader = PdfReader(os.BytesIO(pdf_bytes))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages).strip()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {e}")

    if not text:
        raise HTTPException(status_code=400, detail="Failed to extract text.")

    title = title or file.filename

    # Save doc entry
    doc = IngestedDocument(
        title=title,
        source="pdf",
        original_path=file.filename,
        collection=collection,
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    # Chunk
    chunks = chunk_text(text)
    embeddings = list(embedder.embed(chunks))

    # Store
    for idx, (chunk_text_str, emb) in enumerate(zip(chunks, embeddings)):
        session.add(IngestedChunk(
            document_id=doc.id,
            chunk_index=idx,
            text=chunk_text_str,
            embedding=list(emb),
            collection=collection,
        ))

    session.commit()

    return {
        "status": "ok",
        "document_id": doc.id,
        "chunks": len(chunks)
    }


# ======================================================
# 4) RAG ANSWER (Retrieval-Augmented Generation)
# ======================================================

@app.get("/rag_answer")
def rag_answer(q: str, top_k: int = 5, session: Session = Depends(get_session)):

    q_emb = list(embedder.embed([q]))[0]

    # Fetch chunks
    rows = session.exec(select(IngestedChunk)).all()

    scored = []
    for r in rows:
        emb = np.array(r.embedding)
        sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb)))
        scored.append((sim, r))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[:top_k]

    context = "\n\n".join([f"[Chunk {r.chunk_index}] {r.text}" for (_, r) in top])

    prompt = f"""
Use ONLY the context below to answer the question.

--- CONTEXT ---
{context}

--- QUESTION ---
{q}

Answer clearly and concisely based on the context.
"""

    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    final = resp.choices[0].message.content

    return {
        "query": q,
        "retrieved_chunks": [
            {"score": s, "text": r.text, "chunk_id": r.id}
            for (s, r) in top
        ],
        "final_answer": final
    }


# ======================================================
# 5) STREAMING ENDPOINT
# ======================================================

@app.get("/stream")
async def stream_answer(q: str):
    async def token_stream():
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": q}],
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
                await asyncio.sleep(0.01)
        yield "\n[END]"

    return StreamingResponse(token_stream(), media_type="text/plain")
