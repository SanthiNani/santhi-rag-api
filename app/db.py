from datetime import datetime
from typing import Optional, List, Any

from sqlmodel import SQLModel, Field, create_engine, Session, Column, JSON


DATABASE_URL = "sqlite:///chatlogs.db"

engine = create_engine(DATABASE_URL, echo=False)


class ChatLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str

    # Correct way to store embeddings as JSON
    embedding: Optional[List[float]] = Field(
        sa_column=Column(JSON)
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)


class IngestedDocument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    source: str
    original_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    collection: str = "default"


class IngestedChunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="ingesteddocument.id")
    chunk_index: int
    text: str

    embedding: List[float] = Field(
        sa_column=Column(JSON)
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    collection: str = "default"


def create_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
