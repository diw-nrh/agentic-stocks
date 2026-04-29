from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from sqlmodel import Session
from memory.database import get_session
from memory.skills import service

router = APIRouter(prefix="/skills", tags=["skills"])

# Pydantic Schema สำหรับตรวจสอบข้อมูลขาเข้า (Validation)
class SkillSaveRequest(BaseModel):
    name: str
    description: str
    source_code: str
    tool_schema: Optional[Dict[str, Any]] = None

class SkillSearchRequest(BaseModel):
    query: str
    limit: int = 3

@router.get("/search")
def search_endpoint(req: SkillSearchRequest, session: Session = Depends(get_session)):
    results = service.search_skills(session, req.query, req.limit)
    return {"results": [
        {"name": s.name, "description": s.description, "source_code": s.source_code}
        for s in results
    ]}

@router.get("/all")
def all_endpoint(session: Session = Depends(get_session)):
    results = service.get_all_skills(session)
    return {"results": [
        {"name": s.name, "description": s.description, "source_code": s.source_code}
        for s in results
    ]}

@router.post("/save")
def save_endpoint(req: SkillSaveRequest, session: Session = Depends(get_session)):
    saved = service.save_skill(session, req.model_dump())
    return {"status": "success", "skill_id": saved.id, "skill_name": saved.name}
