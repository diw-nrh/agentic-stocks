#multi agent planner graph
import sys
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.registry import TOOLS_STOCK, TOOLS_WEATHER, get_select_skills_prompt,weather
from shared.config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from state import main_state
from schema import ToTGenerator, ToTEvaluator
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model=Config.AI.MODEL_NAME, 
                   api_key=Config.AI.NOVITA_API_KEY, 
                   base_url=Config.AI.BASE_URL)

def route_to_agents(state: main_state):
    tasks = state.get("agent_tasks", {})
    if not tasks:
        return "fusion_node"
    return list(tasks.keys())

# Planner node
def planner_node(state: main_state):
    user_input = state["messages"][-1].content
    refined_task = llm.invoke(
        "Analyze the following user input and extract a clear, actionable task description for a multi-agent system. "
        "Identify the core objective, specific parameters (e.g., locations, stock symbols, dates), and constraints. "
        "Remove any conversational filler and output only the direct instructions. "
        f"User input: {user_input}"
    ).content
    
    # Generate 3 plan options
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Multi-Agent system planner. You manage the following agents: weather_agent, stocks_agent. "
         "Read the prompt and design 3 different work plans, for example: quantitative-focused, balanced, or data-focused."),
        ("user", "The task is: {input}")
    ])
    
    llm_generator = generator_prompt | llm.with_structured_output(ToTGenerator)
    tot_drafts = llm_generator.invoke({"input": refined_task})
    
    # Evaluate and select the best plan
    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are the head plan evaluator. Read all 3 plan options and select the one that is 'most comprehensive and utilizes the Agents most effectively'."),
        ("user", "User's task: {input}\n\nPlan options:\n{options}")
    ])
    
    llm_evaluator = evaluator_prompt | llm.with_structured_output(ToTEvaluator)
    evaluation_result = llm_evaluator.invoke({
        "input": refined_task,
        "options": tot_drafts.json()
    })
    
    best_plan = next(opt for opt in tot_drafts.options if opt.option_id == evaluation_result.best_option_id)
    
    return {"agent_tasks": best_plan.tasks}

# Specific agent nodes
def weather_agent_node(state: main_state):
    skill_prompt = get_select_skills_prompt("skills/weather")
    messages = [{"role": "system", "content": skill_prompt}]
    weather_task = state.get("agent_tasks", {}).get("weather_agent", "")
    if isinstance(weather_task, str) and weather_task.strip():
        messages.append({"role": "user", "content": weather_task})
    elif isinstance(weather_task, list):
        messages.extend(weather_task)
    llm_weather = create_agent(model=llm, tools=[weather])
    result = llm_weather.invoke({
        "messages": messages})
    return {"agent_results": {"weather": result}}

def stocks_agent_node(state: main_state):
    skill_prompt = get_select_skills_prompt("skills/stocks")
    messages = [{"role": "system", "content": skill_prompt}]
    stocks_task = state.get("agent_tasks", {}).get("stocks_agent", "")
    if isinstance(stocks_task, str) and stocks_task.strip():
        messages.append({"role": "user", "content": stocks_task})
    elif isinstance(stocks_task, list):
        messages.extend(stocks_task)
    llm_stocks = create_agent(model=llm, tools=TOOLS_STOCK)
    result = llm_stocks.invoke({"messages": messages})
    return {"agent_results": {"stocks": result}}

# Fusion and presentation nodes
def fusion_node(state: main_state):
    results = llm.invoke(f"Summarize the following data into a complete report: {state['agent_results']}")
    return {"fusion_results": results}

def presentation_node(state: main_state):
    return {"results": state['fusion_results']}

workflow = StateGraph(main_state)
workflow.add_node("planner", planner_node)
workflow.add_node("weather_agent", weather_agent_node)
workflow.add_node("stocks_agent", stocks_agent_node)
workflow.add_node("fusion", fusion_node)
workflow.add_node("presentation", presentation_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner", 
    route_to_agents,
    {
        "weather_agent": "weather_agent",
        "stocks_agent": "stocks_agent",
        "fusion": "fusion"
    }
)
workflow.add_edge("weather_agent", "fusion")
workflow.add_edge("stocks_agent", "fusion")
workflow.add_edge("fusion", "presentation")
workflow.add_edge("presentation", END)
app = workflow.compile()