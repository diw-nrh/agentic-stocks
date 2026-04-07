#multi agent planner graph
from langgraph.graph import StateGraph,END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from tools.registry import TOOLS
from shared.config import Config
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from state import main_state
from schema import ToTGenerator, ToTEvaluator
llm = ChatOpenAI(model=Config.AI.MODEL_NAME, 
                   api_key=Config.AI.NOVITA_API_KEY, 
                   base_url=Config.AI.BASE_URL)


def route_to_agents(state: main_state):
    tasks = state.get("agent_tasks", {})
    return list(tasks.keys())

# Planner node
def planner_node(state: main_state):
    user_input = state.messages[-1].content
    
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
    prompt =ChatPromptTemplate.from_messages([
        ("system", "คุณคือ weather_agent ที่เชี่ยวชาญในการให้ข้อมูลสภาพอากาศ"),
        ("user", "จงทำตามคำสั่งนี้: {input}")
    ])
    llm_weather = prompt | llm.with_structured_output(str)
    return {"agent_tasks": {**state.agent_tasks, "weather": "weather"}}
def news_agent_node(state: main_state):
    return {"agent_tasks": {**state.agent_tasks, "news": "news"}}
def stocks_agent_node(state: main_state):
    return {"agent_tasks": {**state.agent_tasks, "stocks": "stocks"}}
# Fusion and presentation nodes
def fusion_node(state: main_state):
    return {"agent_tasks": {**state.agent_tasks}}
def presentation_node(state: main_state):
    return {"agent_tasks": {**state.agent_tasks}}

app = StateGraph(main_state)
app.add_node("planner", planner_node)
app.add_node("weather_agent", weather_agent_node)
app.add_node("news_agent", news_agent_node)
app.add_node("stocks_agent", stocks_agent_node)
app.add_node("fusion", fusion_node, tools_condition(TOOLS))
app.add_node("presentation", presentation_node)

app.set_entry_point("planner")
app.add_conditional_edges(
    "planner", 
    route_to_agents,
    {
        "weather_agent": "weather_agent",
        "news_agent": "news_agent",
        "stocks_agent": "stocks_agent",
        "fusion": "fusion"
    }
)
app.add_edge("weather_agent", "fusion")
app.add_edge("news_agent", "fusion")
app.add_edge("stocks_agent", "fusion")
app.add_edge("fusion", "presentation")
app.add_edge("presentation", END)