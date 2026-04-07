import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from tools.registry import TOOLS

load_dotenv()

model = ChatOpenAI(model="deepseek/deepseek-v3.2", 
                   api_key=os.getenv("NOVITA_API_KEY"), 
                   base_url="https://api.novita.ai/openai")

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
    tools_condition,
    {
        "tools": "tools",
        "__end__": END,
    },
)

graph.add_edge("tools", "agent")

app = graph.compile()