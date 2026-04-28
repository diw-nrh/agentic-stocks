import os
import sys
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import text

# Load shared config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import Config

DATABASE_URL = Config.Memory.DATABASE_URL

engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
