#multi agent planner graph
import sys
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.registry import TOOLS_STOCK, TOOLS_WEATHER, get_select_skills_prompt
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
    weather_skill_prompt = get_select_skills_prompt("skills/weather")
    stocks_skill_prompt = get_select_skills_prompt("skills/stocks")

    # แปลงข้อความให้ระบุวัตถุประสงค์ของผู้ใช้ชัดเจนขึ้น
    refined_task = llm.invoke(
        "You are a task router. Analyze the user input and determine which agents are needed.\n\n"
        "## AVAILABLE AGENTS & THEIR TOOLS:\n"
        f"### weather_agent capabilities:\n{weather_skill_prompt}\n\n"
        f"### stocks_agent capabilities:\n{stocks_skill_prompt}\n\n"
        "## YOUR JOB:\n"
        "1. Identify what the user is asking for — weather only, stocks only, or both.\n"
        "2. If weather is needed: extract the specific city/cities mentioned.\n"
        "3. If stocks are needed: extract the specific tickers or companies mentioned.\n"
        "4. ONLY include agents that are directly relevant to the user's request.\n"
        "   - User asks 'weather in Bangkok' → weather_agent only\n"
        "   - User asks 'AAPL stock price' → stocks_agent only\n"
        "   - User asks 'how does rain affect energy stocks' → both agents\n\n"
        "Output a clear, minimal task description for each required agent.\n"
        f"User input: {user_input}"
    ).content

    # 2. ออกแบบแผนมา 3 แบบ ToT
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a Multi-Agent system planner. Design 3 lean work plans based on the task analysis.\n\n"
         "## AGENT CAPABILITIES:\n"
         f"### weather_agent:\n{weather_skill_prompt}\n\n"
         f"### stocks_agent:\n{stocks_skill_prompt}\n\n"
         "## RULES:\n"
         "1. Only assign tasks to agents that are actually needed for the user's request.\n"
         "2. If only weather is needed, plans should only contain 'weather_agent'.\n"
         "3. If only stocks are needed, plans should only contain 'stocks_agent'.\n"
         "4. For stocks_agent instructions: provide ONLY a comma-separated list of tickers, no sector labels.\n"
         "5. Keep scope minimal — do not expand the request beyond what the user asked."),
        ("user", "Task analysis: {input}")
    ])

    llm_generator = generator_prompt | llm.with_structured_output(ToTGenerator)
    tot_drafts = llm_generator.invoke({"input": refined_task})

    # เลือกตัวที่ดีที่สุด
    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a plan evaluator. Select the plan that best matches what the user actually asked for.\n"
         "Prefer plans that are focused and minimal — do NOT favor plans that use more agents or gather more data than necessary."),
        ("user", "User's original request: {input}\n\nPlan options:\n{options}")
    ])

    llm_evaluator = evaluator_prompt | llm.with_structured_output(ToTEvaluator)
    evaluation_result = llm_evaluator.invoke({
        "input": user_input,  # ใช้ user_input ตรงๆ ไม่ใช่ refined_task
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
    llm_weather = create_agent(model=llm, tools=TOOLS_WEATHER)
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
    user_question = state["messages"][-1].content
    prompt = (
        f"You are an expert Data Analyst. The user asked this specific question: '{user_question}'\n\n"
        "YOUR TASK:\n"
        "1. Answer the user's question directly using ONLY the gathered data below.\n"
        "2. DO NOT force a correlation if the user didn't ask for one (e.g., if they only asked for the weather, just report the weather professionally).\n"
        "3. IF the user asked for a relationship or correlation AND the data supports it, explicitly explain the connection.\n\n"
        f"Agent Data: {state['agent_results']}"
    )
    
    results = llm.invoke(prompt).content
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