import sys
import os
from sqlmodel import Session, select, or_
from memory.skills.models import Skill

# Load shared config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared.config import Config
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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

llm_transformer = ChatOpenAI(
    model=Config.AI.MODEL_NAME, 
    api_key=Config.AI.NOVITA_API_KEY, 
    base_url=Config.AI.BASE_URL
)

def transform_query(query: str) -> str:
    """Modular RAG: Transforms a simple user query into a technical query for better vector search."""
    try:
        sys_msg = SystemMessage(content="You are an expert search query optimizer for code. Convert the user's intent into a comma-separated list of technical keywords (e.g., Python functions, API names) suitable for semantic searching of tool source code. Reply ONLY with the keywords.")
        hum_msg = HumanMessage(content=query)
        response = llm_transformer.invoke([sys_msg, hum_msg])
        return response.content.strip()
    except Exception as e:
        print(f"Query Transform Error: {e}")
        return query

def search_skills(session: Session, query: str, limit: int = 3):
    # Modular RAG: Transform Query
    technical_query = transform_query(query)
    query_vector = get_embedding(technical_query)
    
    # Advanced RAG: Hybrid Search approach
    # Phase 1: Try finding skills that share exact keywords AND sort by semantic closeness
    stmt = select(Skill).where(
        or_(
            Skill.name.ilike(f"%{query}%"),
            Skill.description.ilike(f"%{query}%")
        )
    ).order_by(Skill.embedding.cosine_distance(query_vector)).limit(limit)
    
    results = session.exec(stmt).all()
    
    # Phase 2: Fallback to pure semantic search if no keyword matches
    if not results:
        stmt = select(Skill).order_by(Skill.embedding.cosine_distance(query_vector)).limit(limit)
        results = session.exec(stmt).all()
        
    return results

def get_all_skills(session: Session):
    stmt = select(Skill)
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
