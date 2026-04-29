from sqlmodel import create_engine, SQLModel, Session
from shared.config import Config

DATABASE_URL = Config.Memory.DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
