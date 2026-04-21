import sys
import os
from sqlmodel import Session, select
from memory.models import Skill

# Load shared config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.config import Config
from langchain_openai import OpenAIEmbeddings

# Create Embedding Model 
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=Config.AI.NOVITA_API_KEY,
    base_url=Config.AI.BASE_URL
)

def get_embedding(text: str):
    if not text:
        return [0.0] * 1536
    try:
        return embeddings_model.embed_query(text)
    except Exception as e:
        print(f"Generate Embedding Error: {e}")
        return [0.0] * 1536

def search_skills(session: Session, query: str, limit: int = 3):
    query_vector = get_embedding(query)
    
    # Cosine distance to find similar skills in Memory
    stmt = select(Skill).order_by(Skill.embedding.cosine_distance(query_vector)).limit(limit)
    results = session.exec(stmt).all()
    return results

def save_skill(session: Session, skill_data: dict):
    embedding_vector = get_embedding(skill_data.get("description", ""))
    
    db_skill = Skill(
        name=skill_data["name"],
        description=skill_data.get("description", ""),
        tool_schema=skill_data.get("tool_schema", {}),
        source_code=skill_data.get("source_code", ""),
        embedding=embedding_vector
    )
    session.add(db_skill)
    session.commit()
    session.refresh(db_skill)
    return db_skill
