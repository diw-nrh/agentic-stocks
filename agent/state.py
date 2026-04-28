from langgraph.graph import MessagesState
import operator
from typing import Annotated, Any

class main_state(MessagesState):
    agent_results: Annotated[dict[str, Any], operator.ior]
    fusion_results: Any

    # Input validation fields
    input_validated: bool
    gathered_info: Annotated[list, operator.add]  # Clarification Q&A history

    # Dual-Plan fields (Planner v2)
    plan_steps: list       # List of PlanStep dicts
    agent_specs: list      # List of AgentSpec dicts
    plan_ready: bool       # set by plan_observer
    agents_ready: bool     # set by agent_observer
    replan_count: int      # guard against infinite replan loops