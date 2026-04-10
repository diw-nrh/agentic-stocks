from pydantic import BaseModel, Field
from typing import List, Dict

class PlanOption(BaseModel):
    option_id: int
    tasks: Dict[str, str] = Field(description="Use the agent's name as the key (weather_agent, news_agent, stocks_agent) and the prompt/command as the value (Strict rule: If a specific agent is not needed, do not include its key).")
    rationale: str = Field(description="The rationale for why this plan is effective.")

class ToTGenerator(BaseModel):
    options: List[PlanOption] = Field(description="Generate 3 distinct plan options.")

class ToTEvaluator(BaseModel):
    best_option_id: int = Field(description="The ID of the most suitable plan.")
    critique: str = Field(description="The reason for selecting this plan and any potential caveats.")