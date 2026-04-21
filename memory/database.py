import os
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import text

# We default to the internal docker network URL if not provided by the .env
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hermes:hermes_pass@postgres:5432/hermes_db")

engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
