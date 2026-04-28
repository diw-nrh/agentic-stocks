from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class InputValidation(BaseModel):
    is_sufficient: bool = Field(description="Whether the user's input contains all information needed to proceed.")
    reasoning: str = Field(description="Why the input is or isn't sufficient.")
    clarification_question: Optional[str] = Field(
        default=None,
        description="A clear, concise question to ask the user for the specific missing data. Only set when is_sufficient=False."
    )


class PlanStep(BaseModel):
    step_id: int = Field(description="Sequential step number starting from 1.")
    description: str = Field(description="What needs to be done in this step.")
    agent_name: str = Field(description="Name of the agent responsible for this step. Must match a name in agent_specs.")


class AgentSpec(BaseModel):
    name: str = Field(description="Unique agent name. Must exactly match agent_name used in plan_steps.")
    role: str = Field(description="The agent's specialized role.")
    goal: str = Field(description="The specific goal this agent must achieve.")
    tools_required: List[str] = Field(
        description="Tool names this agent needs. Valid names: 'weather', 'stock', 'news', 'manage_skill'."
    )


class DualPlan(BaseModel):
    reasoning: str = Field(description="Step-by-step analysis: what is needed, which agents, and why.")
    plan_steps: List[PlanStep] = Field(description="Ordered list of steps. Each step maps to exactly one agent.")
    agent_specs: List[AgentSpec] = Field(description="One spec per unique agent_name referenced in plan_steps.")