#multi agent planner graph
import sys
import os
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.registry import TOOLS_STOCK, TOOLS_WEATHER, get_select_skills_prompt, weather
from shared.config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from state import main_state
from schema import ToTGenerator, ToTEvaluator
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model=Config.AI.MODEL_NAME, 
                   api_key=Config.AI.NOVITA_API_KEY, 
                   base_url=Config.AI.BASE_URL)

# ==========================================
# 🌟 1. ระบบ Dynamic Routers (สวิตช์สับรางอัจฉริยะ)
# ==========================================
def start_routing(state: main_state):
    mode = state.get("route_mode", "parallel")
    tasks = state.get("agent_tasks", {})
    if not tasks: 
        return "fusion"
    
    if mode == "parallel":
        return list(tasks.keys()) # วิ่งขนาน
    elif mode == "sequential_weather_first" and "weather_agent" in tasks:
        return "weather_agent"    # ไปอากาศก่อน
    elif mode == "sequential_stocks_first" and "stocks_agent" in tasks:
        return "stocks_agent"     # ไปหุ้นก่อน
    
    return list(tasks.keys())     # Fallback: ถ้าเลือกโหมดแปลกๆ ให้รันขนาน

def weather_routing(state: main_state):
    mode = state.get("route_mode", "parallel")
    tasks = state.get("agent_tasks", {})
    # ถ้าอากาศเสร็จแล้ว และอยู่ในโหมดให้อากาศไปก่อน แล้วยังมีคิวของหุ้นเหลืออยู่
    if mode == "sequential_weather_first" and "stocks_agent" in tasks:
        return "stocks_agent" 
    return "fusion" # โหมด parallel หรือไม่มีคิวหุ้นแล้ว ไปสรุปเลย

def stocks_routing(state: main_state):
    mode = state.get("route_mode", "parallel")
    tasks = state.get("agent_tasks", {})
    # ถ้าหุ้นเสร็จแล้ว และอยู่ในโหมดให้หุ้นไปก่อน แล้วยังมีคิวของอากาศเหลืออยู่
    if mode == "sequential_stocks_first" and "weather_agent" in tasks:
        return "weather_agent" 
    return "fusion"


# ==========================================
# 🌟 2. Nodes (ส่วนของ AI แต่ละแผนก)
# ==========================================
def planner_node(state: main_state):
    user_input = state["messages"][-1].content
    
    # Task Extraction: บังคับให้คายแต่ชื่อหุ้นเพียวๆ ห้ามมีหมวดหมู่
    refined_task = llm.invoke(
        "Analyze the user input. Extract the goal and target data. "
        "CRITICAL RULES: "
        "1. If cities are needed, provide 4-5 exact city names. "
        "2. If stocks are needed, generate a FLAT, COMMA-SEPARATED list of 5-8 specific stock tickers. "
        "DO NOT use categories, bullet points, or sector names for stocks (e.g., AAPL, MSFT, TSLA)."
        f"\n\nUser input: {user_input}"
    ).content
    
    # Generator: สอน Planner ให้เลือกโหมดสับรางให้ถูกกับโจทย์
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Master Multi-Agent Planner. You must design 3 plans and decide the BEST execution route.\n\n"
         "### HOW TO CHOOSE THE 'route_mode':\n"
         "1. 'parallel': Use if queries are independent (e.g., 'Weather in NY and Apple stock'). Maximum speed.\n"
         "2. 'sequential_weather_first': Use if stock selection heavily DEPENDS on finding the weather first (e.g., 'What stocks to buy based on the current weather?').\n"
         "3. 'sequential_stocks_first': Use if weather check DEPENDS on the stock (e.g., 'Check Apple stock, then weather at its HQ').\n\n"
         "### INSTRUCTIONS FORMATTING:\n"
         "Provide extremely detailed tool instructions. For stocks_agent, ONLY provide flat ticker symbols (no categories or intro text)."),
        ("user", "Extracted details: {input}")
    ])
    
    llm_generator = generator_prompt | llm.with_structured_output(ToTGenerator)
    tot_drafts = llm_generator.invoke({"input": refined_task})
    
    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", "Select the best plan with the most logical 'route_mode' and cleanest data extraction."),
        ("user", "Task: {input}\n\nPlans:\n{options}")
    ])
    
    llm_evaluator = evaluator_prompt | llm.with_structured_output(ToTEvaluator)
    evaluation_result = llm_evaluator.invoke({"input": refined_task, "options": tot_drafts.json()})
    
    best_plan = next(opt for opt in tot_drafts.options if opt.option_id == evaluation_result.best_option_id)
    
    # คืนค่า agent_tasks ไปให้ Tools และคืนค่า route_mode ให้ Router อ่าน
    return {"agent_tasks": best_plan.tasks, "route_mode": best_plan.route_mode}

def weather_agent_node(state: main_state):
    skill_prompt = get_select_skills_prompt("skills/weather")
    messages = [{"role": "system", "content": skill_prompt}]
    
    weather_task = state.get("agent_tasks", {}).get("weather_agent", "")
    if isinstance(weather_task, str) and weather_task.strip():
        # [การคุยข้ามแผนก] ถ้าหุ้นทำงานมาก่อน เอาข้อมูลหุ้นให้แผนกอากาศอ่านด้วย!
        if state.get("route_mode") == "sequential_stocks_first" and "stocks" in state.get("agent_results", {}):
            weather_task = f"Previous Stock Data:\n{state['agent_results']['stocks']}\n\nYour Task: {weather_task}"
        messages.append({"role": "user", "content": weather_task})
        
    llm_weather = create_agent(model=llm, tools=[weather])
    result = llm_weather.invoke({"messages": messages})
    
    current_results = state.get("agent_results", {}).copy()
    current_results["weather"] = result
    return {"agent_results": current_results}

def stocks_agent_node(state: main_state):
    skill_prompt = get_select_skills_prompt("skills/stocks")
    messages = [{"role": "system", "content": skill_prompt}]
    
    stocks_task = state.get("agent_tasks", {}).get("stocks_agent", "")
    if isinstance(stocks_task, str) and stocks_task.strip():
        # [การคุยข้ามแผนก] ถ้าอากาศทำงานมาก่อน เอาข้อมูลอากาศให้แผนกหุ้นอ่านด้วย!
        if state.get("route_mode") == "sequential_weather_first" and "weather" in state.get("agent_results", {}):
            stocks_task = f"CRITICAL CONTEXT - Real-time Weather Just Gathered:\n{state['agent_results']['weather']}\n\nYour Task: {stocks_task}"
        messages.append({"role": "user", "content": stocks_task})
        
    llm_stocks = create_agent(model=llm, tools=TOOLS_STOCK)
    result = llm_stocks.invoke({"messages": messages})
    
    current_results = state.get("agent_results", {}).copy()
    current_results["stocks"] = result
    return {"agent_results": current_results}

def fusion_node(state: main_state):
    # ย้ำ Fusion อีกรอบว่าให้หา Correlation ถ้าวิ่งแบบต่อเนื่อง
    mode = state.get("route_mode", "parallel")
    prompt = f"Synthesize this data into a complete report. Execution mode was {mode}. If sequential, explicitly explain how the first dataset influenced or correlates with the second. Data: {state['agent_results']}"
    results = llm.invoke(prompt).content # .content เพื่อดึงข้อความออกมา
    return {"fusion_results": results}

def presentation_node(state: main_state):
    return {"results": state['fusion_results']}


# ==========================================
# 🌟 3. Graph Definition (ประกอบร่าง)
# ==========================================
workflow = StateGraph(main_state)
workflow.add_node("planner", planner_node)
workflow.add_node("weather_agent", weather_agent_node)
workflow.add_node("stocks_agent", stocks_agent_node)
workflow.add_node("fusion", fusion_node)
workflow.add_node("presentation", presentation_node)

workflow.set_entry_point("planner")

# เริ่มต้น: ให้ Planner ส่งให้ start_routing เลือกเลน
workflow.add_conditional_edges("planner", start_routing, ["weather_agent", "stocks_agent", "fusion"])

# เมื่อ Agent 1 ทำเสร็จ: ให้ Routing เลือกส่งไม้ต่อให้ Agent 2 หรือไปจบที่ Fusion
workflow.add_conditional_edges("weather_agent", weather_routing, ["stocks_agent", "fusion"])
workflow.add_conditional_edges("stocks_agent", stocks_routing, ["weather_agent", "fusion"])

# จบงาน
workflow.add_edge("fusion", "presentation")
workflow.add_edge("presentation", END)

app = workflow.compile()