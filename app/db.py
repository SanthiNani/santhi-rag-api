from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, create_engine, Session, JSON

DATABASE_URL = "sqlite:///chatlogs.db"

engine = create_engine(DATABASE_URL, echo=False)


class ChatLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str
    embedding: Optional[List[float]] = Field(sa_column=Field(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow)


# NEW: Ingested document metadata
class IngestedDocument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    source: str  # "pdf", "text", "url"
    original_path: Optional[str] = None  # filename or url
    created_at: datetime = Field(default_factory=datetime.utcnow)
    collection: str = "default"  # for grouping docs


# NEW: Ingested chunks table
class IngestedChunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="ingesteddocument.id")
    chunk_index: int
    text: str
    embedding: List[float] = Field(sa_column=Field(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    collection: str = "default"


def create_db():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
