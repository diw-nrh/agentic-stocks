from langgraph.graph import MessagesState
import operator
from typing import Annotated

class main_state(MessagesState):
    agent_tasks: dict[str, str]
    
    agent_results: Annotated[dict[str, str], operator.ior]
    
    fusion_results: Annotated[dict[str, str], operator.ior]
    
    results: str