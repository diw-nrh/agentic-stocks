from langgraph.graph import StateGraph, MessagesState

class main_state(MessagesState):
    agent_tasks: dict[str, str]
