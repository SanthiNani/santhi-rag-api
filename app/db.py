# app/db.py

from datetime import datetime
from typing import Optional, List, Iterator

from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON as SQLITE_JSON

import os

# You can override on Render with env var
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chatlogs.db")

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)


class ChatLog(SQLModel, table=True):
    """
    Stores ad-hoc chat Q&A with embeddings for semantic history search (future use).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str

    # IMPORTANT: Use SQLAlchemy Column(JSON) + Python float list to avoid
    # "Object of type float32 is not JSON serializable" errors.
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(SQLITE_JSON, nullable=True),
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestedDocument(SQLModel, table=True):
    """
    Metadata for each ingested document (text or PDF).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    source: str  # "pdf", "text", "url"
    original_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    collection: str = "default"  # logical grouping (e.g. "resume", "docs")


class IngestedChunk(SQLModel, table=True):
    """
    One row per chunk with its embedding.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="ingesteddocument.id")
    chunk_index: int
    text: str

    embedding: List[float] = Field(
        sa_column=Column(SQLITE_JSON, nullable=False),
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    collection: str = "default"


def create_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
