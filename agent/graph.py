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

    # 2. ออกแบบแผนมา 3 แบบ ToT
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a Multi-Agent system planner. Design 3 lean work plans based on the task analysis.\n\n"

        "## RULES:\n"
        "1. Only assign tasks to agents that are actually needed for the user's request.\n"
        "2. stocks_agent task MUST always include fetching real price data for specific tickers. "
            "Never instruct it to do analysis only without calling stock tools.\n"
        "3. If only weather is needed, plans should only contain 'weather_agent'.\n"
        "4. If only stocks are needed, plans should only contain 'stocks_agent'.\n"
        "5. Keep scope minimal — do not expand the request beyond what the user asked.\n"
        "6. If both agents are needed, each task must be self-contained and independent.\n"
        "   Give stocks_agent a broad instruction covering likely scenarios,\n"
        "   so fusion can match results later. Never reference the other agent's output.\n"
        "   Bad: 'find tickers based on the identified weather' → "
        "   Good: 'fetch tickers for sectors commonly affected by U.S. weather conditions'\n"),
        ("user", "Task analysis: {input}")
    ])

    llm_generator = generator_prompt | llm.with_structured_output(ToTGenerator)
    tot_drafts = llm_generator.invoke({"input": user_input})

    # เลือกตัวที่ดีที่สุด
    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a plan evaluator. Select the plan that best covers the user's question.\n"
        "Prefer plans that gather all information needed to answer completely and accurately.\n"
        "Penalize plans that are missing data sources the user's question depends on.\n"
        "Only avoid redundancy — never sacrifice coverage for brevity."),
        ("user", "User's original request: {input}\n\nPlan options:\n{options}")
    ])

    llm_evaluator = evaluator_prompt | llm.with_structured_output(ToTEvaluator)
    evaluation_result = llm_evaluator.invoke({
        "input": user_input,
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
    stocks_task = state.get("agent_tasks", {}).get("stocks_agent", "")

    # --- Pre-processor: สรุปว่าควรดูหุ้นตัวไหน ---
    ticker_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a financial analyst. Given a task, list the specific stock tickers that should be fetched and investigated.\n"
        "Always return actual ticker symbols (e.g. HD, KMI, NEE). Be concise and direct. Write in plain text."),
        ("user", "Task: {task}")
    ])
    
    ticker_summary = (ticker_prompt | llm).invoke({"task": stocks_task})
    enriched_task = f"{stocks_task}\n\nStock focus: {ticker_summary.content}"
    
    print("=== enriched_task ===")  # ← เพิ่มตรงนี้
    print(enriched_task)
    # --- Stocks Agent ---
    skill_prompt = get_select_skills_prompt("skills/stock")
    system = (
        "stock agent skill tool: " + skill_prompt +
        "\nComplete the task autonomously. Never ask the user for input."
        "\nWhen fetching stock data, you MUST call the stock tool ONE ticker at a time. "
        "Never pass multiple tickers in a single call. "
        "Example: call stock('HD'), then stock('LOW'), then stock('NEE') separately."
        "\nWhen returning results, always include the ticker symbol with each data point. "
        "Format: TICKER: <price info>."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": enriched_task}
]
    llm_stocks = create_agent(model=llm, tools=TOOLS_STOCK)
    result = llm_stocks.invoke({"messages": messages})
    return {"agent_results": {"stocks": result}}

# Fusion and presentation nodes
def fusion_node(state: main_state):
    user_question = state["messages"][-1].content
    prompt = (
        f"You are an expert analyst. Answer the user's question using the data below.\n\n"
        f"Question: {user_question}\n\n"
        f"Data: {state['agent_results']}"
    )
    
    results = llm.invoke(prompt)
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