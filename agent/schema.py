from pydantic import BaseModel, Field
from typing import List, Dict
class SinglePlan(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis of the user request and why these agents were chosen.")
    tasks: Dict[str, str] = Field(description="Dictionary where keys are agent names and values are their specific tasks.")