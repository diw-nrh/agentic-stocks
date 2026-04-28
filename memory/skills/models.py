from typing import Optional, Any, Dict
from datetime import datetime
from sqlmodel import Field, SQLModel, Column, JSON
from pgvector.sqlalchemy import Vector

class SkillBase(SQLModel):
    name: str = Field(index=True, unique=True, description="The unique name of the skill")
    description: str = Field(description="A human-readable description of what this skill does")
    tool_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    source_code: str = Field(description="The executable python code for this skill")
    success_rate: float = Field(default=0.0)
    call_count: int = Field(default=0)
    status: str = Field(default="ACTIVE")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class Skill(SkillBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    # 1536 dimensions is standard for text-embedding-ada-002 or text-embedding-3-small
    embedding: Optional[Any] = Field(default=None, sa_column=Column(Vector(1536)))
