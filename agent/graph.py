from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from tools.registry import TOOLS
from shared.config import Config

model = ChatOpenAI(model=Config.MODEL_NAME, 
                   api_key=Config.NOVITA_API_KEY, 
                   base_url=Config.BASE_URL)

llm_with_tools = model.bind_tools(TOOLS)

def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    print(f"DEBUG - Tool Calls: {response.tool_calls}") 
    return {"messages": [response]}

tool_node = ToolNode(TOOLS)

graph = StateGraph(MessagesState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    tools_condition
)

graph.add_edge("tools", "agent")

app = graph.compile()