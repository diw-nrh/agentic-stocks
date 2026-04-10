#multi agent planner graph
import sys
import os
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.registry import TOOLS,weather,stock
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
    
    #สร้างทางเลือกแผนงาน 3 แบบ
    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "คุณคือนักวางแผนระบบ Multi-Agent คุณมีลูกน้องคือ weather_agent, news_agent, stocks_agent "
         "จงอ่านโจทย์และออกแบบแผนการทำงานมา 3 รูปแบบที่แตกต่างกัน เช่น แบบเน้นข่าว, แบบเน้นตัวเลข, แบบสมดุล"),
        ("user", "โจทย์คือ: {input}")
    ])
    
    llm_generator = generator_prompt | llm.with_structured_output(ToTGenerator)
    tot_drafts = llm_generator.invoke({"input": user_input})
    # ประเมินและเลือกแผนที่ดีที่สุด
    evaluator_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "คุณคือหัวหน้าผู้ประเมินแผนงาน จงอ่านทางเลือกแผนงานทั้ง 3 แบบ แล้วเลือกแบบที่ 'ครอบคลุมโจทย์ที่สุดและใช้ Agent ได้คุ้มค่าที่สุด'"),
        ("user", "โจทย์ของผู้ใช้: {input}\n\nทางเลือกแผนงาน:\n{options}")
    ])
    
    llm_evaluator = evaluator_prompt | llm.with_structured_output(ToTEvaluator)
    evaluation_result = llm_evaluator.invoke({
        "input": user_input,
        "options": tot_drafts.json()
    })
    
    best_plan = next(opt for opt in tot_drafts.options if opt.option_id == evaluation_result.best_option_id)
    
    return {"agent_tasks": best_plan.tasks}

#  Specific agent nodes
def weather_agent_node(state: main_state):
    
    llm_weather = create_agent(model=llm, tools=[weather])
    result = llm_weather.invoke({
        "messages": state["agent_tasks"].get("weather_agent")})
    return {"agent_results": {"weather": result}}

def news_agent_node(state: main_state):
    llm_news = create_agent(model=llm, tools=TOOLS)
    result = llm_news.invoke({
        "messages": state["agent_tasks"].get("news_agent")})
    return {"agent_results": {"news": result}}

def stocks_agent_node(state: main_state):
    llm_stocks = create_agent(model=llm, tools=[stock])
    result = llm_stocks.invoke({
        "messages": state["agent_tasks"].get("stocks_agent")})
    return {"agent_results": {"stocks": result}}

# Fusion and presentation nodes
def fusion_node(state: main_state):
    results = llm.invoke(f"จงสรุปข้อมูลต่อไปนี้ให้เป็นรายงานฉบับสมบูรณ์: {state['agent_results']}")
    return {"fusion_results": results}

def presentation_node(state: main_state):
    return {"results": state['fusion_results']}

workflow = StateGraph(main_state)
workflow.add_node("planner", planner_node)
workflow.add_node("weather_agent", weather_agent_node)
workflow.add_node("news_agent", news_agent_node)
workflow.add_node("stocks_agent", stocks_agent_node)
workflow.add_node("fusion", fusion_node)
workflow.add_node("presentation", presentation_node)

workflow.set_entry_point("planner")
workflow.add_conditional_edges(
    "planner", 
    route_to_agents,
    {
        "weather_agent": "weather_agent",
        "news_agent": "news_agent",
        "stocks_agent": "stocks_agent",
        "fusion": "fusion"
    }
)
workflow.add_edge("weather_agent", "fusion")
workflow.add_edge("news_agent", "fusion")
workflow.add_edge("stocks_agent", "fusion")
workflow.add_edge("fusion", "presentation")
workflow.add_edge("presentation", END)
app = workflow.compile()