from sqlmodel import SQLModel, Field, create_engine, Session, Column
from datetime import datetime
from typing import List, Optional
from datetime import datetime
from sqlalchemy.types import JSON


class ChatLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    question: str
    answer: str
    embedding: List[float] | None = Field(sa_column=Column(JSON)) 
    created_at: datetime = Field(default_factory=datetime.utcnow)

engine = create_engine("sqlite:///chatlogs.db")

def create_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    return Session(engine)
