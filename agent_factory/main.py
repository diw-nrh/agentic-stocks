import json
from typing import TypedDict, List, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# --- 1. Define the State (หน่วยความจำของโรงงาน) ---
class AgentBlueprint(TypedDict):
    name: str
    role: str
    goal: str
    tools: List[str]

class AutoAgentState(TypedDict):
    query: str
    team_design: List[AgentBlueprint] # เก็บรายชื่อพนักงานที่ Planner สร้าง
    final_report: List[str]
    current_worker_idx: int

# --- 2. The Planner Node (The Architect) ---
# ทำหน้าที่วิเคราะห์คำสั่งและ "ปั๊ม" รายชื่อ Agent ออกมา
def planner_node(state: AutoAgentState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # ใช้รุ่นฉลาดสุดเป็นตัวแม่
    
    prompt = f"""คุณคือ Meta-Agent Planner หน้าที่ของคุณคือออกแบบ 'ทีม AI' เพื่อแก้โจทย์: {state['query']}
    จงตอบกลับเป็น JSON List ของ Agent โดยแต่ละตัวต้องมี:
    - name: ชื่อเรียก
    - role: บทบาทเฉพาะทาง
    - goal: สิ่งที่ต้องทำ
    - tools: เครื่องมือที่ต้องใช้ (เช่น 'search', 'calculator', 'stock_api')
    
    สร้างทีมที่เหมาะสม (ไม่เกิน 3 ตัว) และเรียงลำดับการทำงานที่ถูกต้อง
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    # สมมติว่า LLM คืนค่าเป็น JSON string (ในงานจริงควรใช้ JSON Output Parser)
    # ลบ markdown tag ออกถ้ามี
    clean_json = response.content.replace("```json", "").replace("```", "").strip()
    team_design = json.loads(clean_json)
    
    return {"team_design": team_design, "current_worker_idx": 0}

# --- 3. The Dynamic Worker Node (The Factory Machine) ---
# ทำหน้าที่ "กลายร่าง" ตามพิมพ์เขียวที่ได้รับมาทีละตัว
def worker_node(state: AutoAgentState):
    idx = state['current_worker_idx']
    blueprint = state['team_design'][idx]
    
    print(f"--- 🛠️ Factory is creating & running: {blueprint['name']} ({blueprint['role']}) ---")
    
    worker_llm = ChatOpenAI(model="gpt-4o-mini") # ใช้รุ่นเล็ก/เร็วเป็น Worker
    
    system_prompt = f"คุณคือ {blueprint['name']} บทบาทคือ {blueprint['role']} เป้าหมายหลักคือ {blueprint['goal']}"
    
    # รันงาน (ในงานจริงตรงนี้จะมีการเรียก Tools ตาม blueprint['tools'])
    response = worker_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"จากโจทย์หลัก '{state['query']}' โปรดทำหน้าที่ของคุณ")
    ])
    
    return {
        "final_report": state.get('final_report', []) + [f"{blueprint['name']}: {response.content}"],
        "current_worker_idx": idx + 1
    }

# --- 4. Logic สำหรับตรวจสอบว่างานในสายพานหมดหรือยัง ---
def route_next(state: AutoAgentState):
    if state['current_worker_idx'] < len(state['team_design']):
        return "worker"
    return END

# --- 5. Build the Graph ---
workflow = StateGraph(AutoAgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("worker", worker_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "worker")
workflow.add_conditional_edges("worker", route_next)

app = workflow.compile()

# --- ทดลองรันระบบ ---
# ตัวอย่าง Multi-task: หาหุ้น + วิเคราะห์อากาศ
query = "วิเคราะห์ความเสี่ยงของหุ้นสายการบิน Delta หากเกิดพายุเฮอริเคนในแอตแลนติกสัปดาห์หน้า"
for output in app.stream({"query": query}):
    print(output)