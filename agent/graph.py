#multi agent planner graph
import sys
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.registry import TOOLS_STOCK, TOOLS_WEATHER, get_select_skills_prompt,TOOLS_NEWS
from shared.config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from state import main_state
from schema import SinglePlan
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

    cot_planner_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert Multi-Agent Planner. Analyze the user's request step-by-step to create a single, efficient work plan.\n\n"
         
         "## AVAILABLE AGENTS & CAPABILITIES:\n"
         "1. **weather_agent**\n"
         "   - Focus: Meteorological data and climate events.\n"
         "   - Capabilities: Fetches current weather conditions, forecasts, and weather-related news (e.g., storms, floods).\n"
         "   - Constraints: Cannot analyze financial impacts or stock prices.\n"
         "2. **stocks_agent**\n"
         "   - Focus: Financial markets and company data.\n"
         "   - Capabilities: Fetches real-time stock prices, historical data, and financial/business news.\n"
         "   - Constraints: Requires explicit stock tickers for price data. Cannot fetch weather information.\n\n"

         "## PLANNING PROCESS (Chain of Thought):\n"
         "1. **Analyze Intent**: What is the user specifically asking for?\n"
         "2. **Task Decomposition**: Based on the Agents' capabilities, which agent should handle which part of the request?\n"
         "3. **Define Tasks**: Create self-contained tasks that strictly align with each agent's constraints.\n\n"
         
         "## RULES:\n"
         "1. Only assign tasks to agents actually needed. If an agent's capability isn't required, ignore it.\n"
         "2. 'stocks_agent' MUST always be instructed to fetch real price data for specific tickers when discussing stocks.\n"
         "3. If both agents are used, instructions must be independent (No cross-referencing).\n"
         "4. Keep the scope minimal - no extra data unless requested.\n\n"
         
         "Output your internal reasoning first, then the final tasks list."),
        ("user", "Task analysis: {input}")
    ])

    planner = cot_planner_prompt | llm.with_structured_output(SinglePlan)

    plan = planner.invoke({"input": user_input})

    # ปริ้น reasoning ออกมาดูเพื่อตรวจสอบความฉลาดของ Planner
    print(f"=== Planner Reasoning ===\n{plan.reasoning}")

    return {"agent_tasks": plan.tasks}

# Specific agent nodes
def weather_agent_node(state: main_state):
    weather_task = state.get("agent_tasks", {}).get("weather_agent", "")

    # ดึง Skill Prompt มาแยกกัน เพื่อจัด Format ให้เป็นระเบียบ
    skill_prompt_weather = get_select_skills_prompt("skills/weather")
    skill_prompt_news = get_select_skills_prompt("skills/news")
    
    system = (
        f"## WEATHER TOOL DOCUMENTATION:\n{skill_prompt_weather}\n\n"
        f"## NEWS TOOL DOCUMENTATION:\n{skill_prompt_news}\n\n"
        "You are a specialized Weather & Climate Data Agent. Your goal is to gather accurate weather data and relevant news.\n\n"
        "## WORKFLOW & RULES:\n"
        "1. **Weather News (Context)**:\n"
        "   - If the task asks for weather events, disasters (e.g., hurricanes, floods), or climate news, use the `news` tool.\n"
        "   - ALWAYS set `news='weather'`.\n"
        "   - Example: `news(news='weather', location='america', period='current')`\n"
        "2. **Meteorological Data (Conditions/Forecasts)**:\n"
        "   - To fetch exact temperatures, current conditions, or specific forecasts, use the `weather` tool.\n"
        "3. **Execution**:\n"
        "   - You can use both tools if the task requires both context (news) and specific data (weather).\n"
        "   - Summarize your findings clearly."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": str(weather_task)}
    ]
    
    # ใส่ Tools ให้ครบทั้ง 2 ตัว
    llm_weather = create_agent(model=llm, tools=[TOOLS_WEATHER, TOOLS_NEWS])
    result = llm_weather.invoke({"messages": messages})
    return {"agent_results": {"weather": result}}

def stocks_agent_node(state: main_state):
    stocks_task = state.get("agent_tasks", {}).get("stocks_agent", "")

    skill_prompt_stock = get_select_skills_prompt("skills/stock")
    skill_prompt_news = get_select_skills_prompt("skills/news")
    
    system = (
        f"## STOCK TOOL DOCUMENTATION:\n{skill_prompt_stock}\n\n"
        f"## NEWS TOOL DOCUMENTATION:\n{skill_prompt_news}\n\n"
        "You are a specialized Financial Data & News Agent. Your goal is to gather comprehensive information to complete the task.\n\n"
        "## WORKFLOW & RULES:\n"
        "1. **Gather Context (News First)**:\n"
        "   - If the task requires understanding market sentiment or events, use the `news` tool first.\n"
        "   - ALWAYS set `news='stock'` when looking for financial news.\n"
        "   - Example: `news(news='stock', location='america', period='current')`\n"
        "2. **Fetch Data (Stock Prices)**:\n"
        "   - After understanding the context (or if only prices are needed), identify relevant stock tickers.\n"
        "   - Call the `stock` tool for EACH ticker one by one (Never pass multiple symbols in one call).\n"
        "   - Example: `stock(symbols='AAPL', period='current')` then `stock(symbols='MSFT', period='current')`\n"
        "3. **Synthesize & Output**:\n"
        "   - Combine the insights from the news and the real-time stock prices.\n"
        "   - Format your output clearly. Always include TICKER: <data> when discussing specific stocks."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": stocks_task}
    ]
    llm_stocks = create_agent(model=llm, tools=[TOOLS_STOCK, TOOLS_NEWS])
    result = llm_stocks.invoke({"messages": messages})
    return {"agent_results": {"stocks": result}}

# Fusion and presentation nodes
def fusion_node(state: main_state):
    user_question = state["messages"][-1].content
    raw_data = state.get('agent_results', {})

    prompt = f"""You are an expert financial and data analyst. 
        Your MOST IMPORTANT rule: Answer the user's question IMMEDIATELY in the very first sentence, then provide the supporting details.

        ## USER QUESTION:
        "{user_question}"

        ## RAW DATA:
        {raw_data}

        ## STRICT OUTPUT FORMAT:
        **1. Direct Answer:**
        - Start immediately with the final conclusion or recommendation answering the user's question. 
        - Do NOT use filler intros like "Based on the data..." or "Here is the analysis...".
        - Example: "You should monitor utility stocks like EXC and agricultural stocks like ADM due to the extreme cold fronts in the Midwest."

        **2. Explanation (The 'Why'):**
        - Provide 2-3 concise bullet points explaining the logical connection between the weather events and the stock sectors.

        **3. Data Snapshot:**
        - List the fetched tickers with their current price and a 5-word micro-summary.
        - Example: "EXC ($48.57): Cold weather increases heating demand."

        **Constraints:** Keep the entire response under 150-200 words. Be direct, professional, and analytical.
        """
    
    results = llm.invoke(prompt)
    return {"fusion_results": results}


workflow = StateGraph(main_state)
workflow.add_node("planner", planner_node)
workflow.add_node("weather_agent", weather_agent_node)
workflow.add_node("stocks_agent", stocks_agent_node)
workflow.add_node("fusion", fusion_node)

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
workflow.add_edge("fusion", END)
app = workflow.compile()