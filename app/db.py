# app/db.py
from typing import Optional, List
from datetime import datetime

from sqlmodel import SQLModel, Field, create_engine, Session
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON


DATABASE_URL = "sqlite:///./chatlogs.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},  # required for SQLite + FastAPI
)


class ChatLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    question: str
    answer: str
    # store embedding as JSON array of floats
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(JSON)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


def create_db() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    # use: with get_session() as session:
    return Session(engine)
