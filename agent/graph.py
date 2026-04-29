# Multi-agent planner graph
import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as tool_decorator
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from schema import DualPlan, InputValidation
from shared.config import Config
from skills.registry import TOOLS_MANAGE, TOOLS_SEARCH
from state import main_state
from tools.registry import TOOLS_NEWS, TOOLS_STOCK, TOOLS_WEATHER, get_select_skills_prompt

llm = ChatOpenAI(
    model=Config.AI.MODEL_NAME,
    api_key=Config.AI.NOVITA_API_KEY,
    base_url=Config.AI.BASE_URL,
)

# Maps tool name (as declared in AgentSpec) → actual tool object
TOOL_REGISTRY = {
    "weather":      TOOLS_WEATHER,
    "news":         TOOLS_NEWS,
    "stock":        TOOLS_STOCK,
    "manage_skill": TOOLS_MANAGE,
    "search_skill": TOOLS_SEARCH,
}

# Maps tool name → skills doc directory (for building agent system prompts)
SKILL_DOC_MAP = {
    "weather": "skills/weather",
    "news":    "skills/news",
    "stock":   "skills/stock",
}


# ── Helper ───────────────────────────────────────────────────────────────────

def _search_and_register_skill(query: str) -> str | None:
    """Search DB for an existing skill matching *query*, exec its code, and
    register it in TOOL_REGISTRY.  Returns the tool name on success, else None."""
    search_output = TOOLS_SEARCH.invoke({"query": query, "limit": 1})

    name_match = re.search(r"- Name: (.+)", search_output)
    code_start = search_output.find("  Code: ")

    if not name_match or code_start == -1 or "No matching skills found" in search_output:
        return None

    tool_name = name_match.group(1).strip()
    source_code = search_output[code_start + len("  Code: "):]

    try:
        namespace = {"tool": tool_decorator, "__builtins__": __builtins__}
        exec(source_code, namespace)  # noqa: S102
        for value in namespace.values():
            if hasattr(value, "name") and hasattr(value, "invoke") and value is not tool_decorator:
                TOOL_REGISTRY[tool_name] = value
                return tool_name
    except Exception as exc:
        print(f"[skill register] exec failed: {exc}")

    return None

def input_validator_node(state: main_state):
    messages = state["messages"]

    validator_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an Input Validation Agent. Think step by step before deciding.\n\n"
         "## CHAIN OF THOUGHT — follow these steps in order:\n"
         "STEP 1: What is the user ultimately trying to accomplish?\n"
         "STEP 2: Is there a HARD BLOCKER — a piece of data so fundamental that no agent could even START without it?\n"
         "         A hard blocker is something that cannot be inferred, researched, or decided by a downstream agent.\n"
         "STEP 3: If YES hard blocker → ask ONE short question for ONLY that missing data. Set is_sufficient=False.\n"
         "         If NO hard blocker → set is_sufficient=True. Trust the planner and agents to handle everything else.\n\n"
         "## WHAT COUNTS AS A HARD BLOCKER (ask these):\n"
         "- BMI calculation with NO height and NO weight provided (math is impossible without numbers).\n"
         "- Weather request with NO location at all — not even a country or region.\n"
         "- Currency conversion with NO amount specified.\n\n"
         "## WHAT IS NOT A HARD BLOCKER (do NOT ask these):\n"
         "- Which sector/industry to focus on → planner will decide the strategy.\n"
         "- Which specific stocks to analyze → agents can research and pick.\n"
         "- Risk tolerance or investment style → agents can give a general answer.\n"
         "- How to define 'best' or 'worth buying' → agents handle open-ended analysis.\n"
         "- Any detail the agent could look up, infer, or decide independently.\n\n"
         "## EXAMPLES:\n"
         "- 'What stocks fit the weather in Thailand?' → NO hard blocker. is_sufficient=True.\n"
         "- 'What should I invest in?' → NO hard blocker. is_sufficient=True.\n"
         "- 'Compare tech stocks' → NO hard blocker. is_sufficient=True.\n"
         "- 'What is the weather?' → HARD BLOCKER (no location). Ask for location.\n"
         "- 'Calculate my BMI' → HARD BLOCKER (no height/weight). Ask for them.\n\n"
         "When in doubt, default to is_sufficient=True."),
        ("user", "Conversation so far:\n{conversation}")
    ])

    validator = validator_prompt | llm.with_structured_output(InputValidation)

    conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    ])

    validation = validator.invoke({"conversation": conversation})
    print(f"=== Input Validator ===\nSufficient: {validation.is_sufficient}\nReason: {validation.reasoning}")

    if not validation.is_sufficient:
        user_response = interrupt({"question": validation.clarification_question})

        return {
            "messages": [
                AIMessage(content=validation.clarification_question),
                HumanMessage(content=str(user_response))
            ],
            "gathered_info": [
                {"role": "assistant", "content": validation.clarification_question},
                {"role": "user", "content": str(user_response)}
            ],
            "input_validated": False,
        }

    return {"input_validated": True}


def route_after_validation(state: main_state):
    if state.get("input_validated", False):
        return "planner"
    return "input_validator"


# Planner node
def planner_node(state: main_state):
    full_conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["messages"]
    ])

    tool_list_lines = []
    for tool_name, tool_obj in TOOL_REGISTRY.items():
        description = getattr(tool_obj, "description", "") or ""
        first_line = description.strip().splitlines()[0] if description.strip() else "(no description)"
        tool_list_lines.append(f"- {tool_name:<15}: {first_line}")
    tool_list_str = "\n".join(tool_list_lines)

    cot_planner_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert Meta-Agent Planner. Output a PLAN and AGENT SPECS for every request.\n\n"

         f"## AVAILABLE TOOLS:\n{tool_list_str}\n\n"

         "## RULES:\n"
         "1. Each plan step → exactly ONE agent. agent_name must match exactly in agent_specs.\n"
         "2. tools_required must only contain tool names from AVAILABLE TOOLS above.\n"
         "3. tools_required CAN be empty [] for pure analysis/reasoning agents that need no external data.\n"
         "4. Agents in later steps AUTOMATICALLY receive results from all previous steps as context.\n"
         "   So a later step does NOT need to re-fetch data already retrieved in an earlier step.\n"
         "5. ALWAYS split multi-part requests into multiple steps. Never collapse into 1 step if multiple things are needed.\n"
         "6. For analysis that combines data sources: fetch in early steps, reason/analyze in later steps.\n"
         "7. For ANY calculation or formula-based task (BMI, compound interest, unit conversion, etc.):\n"
         "   ALWAYS create a reusable tool first with manage_skill, then use it in the next step.\n\n"

         "## EXAMPLE — 'Weather in Thailand + stocks worth buying based on weather':\n"
         "  Step 1: WeatherAgent   tools=[weather]  — fetch current weather in Thailand\n"
         "  Step 2: AnalystAgent   tools=[stock]    — use weather context from step 1 to find & evaluate relevant stocks\n\n"

         "## EXAMPLE — 'Compare AAPL and MSFT':\n"
         "  Step 1: StockFetcher  tools=[stock]  — fetch AAPL and MSFT data\n"
         "  Step 2: Analyst       tools=[]       — compare and summarize\n\n"

         "## EXAMPLE — 'Calculate BMI / any formula-based task':\n"
         "  Step 1: SkillMaker    tools=[manage_skill]  — create a Python @tool for the calculation\n"
         "  Step 2: Calculator    tools=[<tool_name>]   — call the created tool with the user's values\n\n"

         "Output reasoning (2-3 sentences), then the structured DualPlan."),
        ("user", "Full conversation:\n{input}")
    ])

    planner = cot_planner_prompt | llm.with_structured_output(DualPlan)
    plan = planner.invoke({"input": full_conversation})

    print(f"=== Planner Reasoning ===\n{plan.reasoning}")
    print(f"=== Plan Steps ===\n{[s.model_dump() for s in plan.plan_steps]}")
    print(f"=== Agent Specs ===\n{[a.model_dump() for a in plan.agent_specs]}")

    return {
        "plan_steps": [s.model_dump() for s in plan.plan_steps],
        "agent_specs": [a.model_dump() for a in plan.agent_specs],
        "plan_ready": False,
        "agents_ready": False,
        "replan_count": state.get("replan_count", 0) + 1,
    }

# ── Observer nodes ──────────────────────────────────────────────────────────

def plan_observer_node(state: main_state):
    """ตรวจสอบความถูกต้องของ plan_steps กับ agent_specs"""
    plan_steps = state.get("plan_steps", [])
    agent_specs = state.get("agent_specs", [])

    if not plan_steps or not agent_specs:
        print("=== Plan Observer === INVALID: empty plan or specs → replan")
        return {"plan_ready": False}

    spec_names = {spec["name"] for spec in agent_specs}
    for step in plan_steps:
        if step["agent_name"] not in spec_names:
            print(f"=== Plan Observer === INVALID: step {step['step_id']} references unknown agent '{step['agent_name']}' → replan")
            return {"plan_ready": False}

    print(f"=== Plan Observer === OK: {len(plan_steps)} steps, {len(agent_specs)} agent specs")
    return {"plan_ready": True}


def agent_observer_node(state: main_state):
    """ตรวจสอบว่า agent_specs ครบถ้วน — unknown tools จะถูกสร้างโดย action_observer อัตโนมัติ"""
    agent_specs = state.get("agent_specs", [])

    for spec in agent_specs:
        if not spec.get("role") or not spec.get("goal"):
            print(f"=== Agent Observer === INVALID: spec '{spec['name']}' missing role/goal → replan")
            return {"agents_ready": False}
        unknown = [t for t in spec.get("tools_required", []) if t not in TOOL_REGISTRY]
        if unknown:
            print(f"=== Agent Observer === NOTE: '{spec['name']}' uses unknown tools {unknown} — will be created at runtime")

    print(f"=== Agent Observer === OK: all {len(agent_specs)} specs valid")
    return {"agents_ready": True}


def action_observer_node(state: main_state):
    """Dynamic agent runner — search skill inline, route to skill_maker if not found"""
    plan_steps = sorted(state.get("plan_steps", []), key=lambda s: s["step_id"])
    specs_by_name = {spec["name"]: spec for spec in state.get("agent_specs", [])}
    current_step_idx = state.get("current_step_idx", 0)

    full_conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["messages"]
    ])

    # โหลด agent_results สะสมจาก state (อาจมีผลจาก round ก่อนหน้า)
    collected_results = dict(state.get("agent_results") or {})

    for i, step in enumerate(plan_steps[current_step_idx:], start=current_step_idx):
        agent_name = step["agent_name"]
        spec = specs_by_name.get(agent_name)
        if not spec:
            continue

        print(f"=== Action Observer === Step {step['step_id']}: running '{agent_name}'")

        # ── If the step needs manage_skill → search DB first ────────────────
        if "manage_skill" in spec.get("tools_required", []):
            search_output = TOOLS_SEARCH.invoke({"query": step["description"], "limit": 1})
            print(f"=== Action Observer === search_skill: {search_output[:120]}...")

            if "No matching skills found" not in search_output:
                registered_name = _search_and_register_skill(step["description"])
                if registered_name:
                    print(f"=== Action Observer === Reused '{registered_name}' from DB")
                    collected_results[agent_name] = search_output
                    continue

            # Not found in DB → route to skill_maker to create it
            print(f"=== Action Observer === '{agent_name}' needs new skill → routing to skill_maker")
            return {
                "agent_results": collected_results,
                "current_step_idx": i,
                "pending_skill_step": step,
            }
            continue

        # Resolve tools — skip unknown ones
        missing_tools = [t for t in spec["tools_required"] if t not in TOOL_REGISTRY]
        if missing_tools:
            print(f"=== Action Observer === WARNING: missing tools {missing_tools} for '{agent_name}' — skipping")
            continue
        tools_for_agent = [TOOL_REGISTRY[t] for t in spec["tools_required"]]

        # Build skill documentation for tools that have SKILL.md
        skill_docs_parts = []
        for t in spec["tools_required"]:
            if t in SKILL_DOC_MAP:
                doc = get_select_skills_prompt(SKILL_DOC_MAP[t])
                if doc:
                    skill_docs_parts.append(f"## {t.upper()} TOOL DOCUMENTATION:\n{doc}")
        skill_docs = "\n\n".join(skill_docs_parts)

        system = (
            (f"{skill_docs}\n\n" if skill_docs else "") +
            f"You are {spec['role']}.\n"
            f"Your goal: {spec['goal']}\n\n"
            f"## YOUR TASK:\n{step['description']}"
        )

        # Pass results from previous steps as context
        prev_results_str = ""
        if collected_results:
            parts = [f"### {n}:\n{r}" for n, r in collected_results.items()]
            prev_results_str = "\n\n## RESULTS FROM PREVIOUS STEPS (use as input context):\n" + "\n\n".join(parts)

        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": full_conversation + prev_results_str},
        ]

        try:
            agent = create_agent(model=llm, tools=tools_for_agent)
            result = agent.invoke({"messages": messages})
            collected_results[agent_name] = result
            print(f"=== Action Observer === Step {step['step_id']}: '{agent_name}' done")
        except Exception as e:
            print(f"=== Action Observer === Step {step['step_id']}: ERROR in '{agent_name}': {e}")
            collected_results[agent_name] = f"ERROR: {e}"

    return {"agent_results": collected_results, "current_step_idx": len(plan_steps), "pending_skill_step": {}}


def skill_maker_node(state: main_state):
    """สร้าง tool ใหม่ บันทึกลง DB และ register เข้า TOOL_REGISTRY แล้ว route กลับไป action_observer"""
    step = state.get("pending_skill_step", {})
    agent_name = step.get("agent_name", "unknown")
    print(f"=== Skill Maker === Step {step.get('step_id')}: creating tool for '{agent_name}'")

    collected_results = dict(state.get("agent_results") or {})
    prev_results_str = ""
    if collected_results:
        parts = [f"### {n}:\n{r}" for n, r in collected_results.items()]
        prev_results_str = "\n\n## RESULTS FROM PREVIOUS STEPS:\n" + "\n\n".join(parts)

    skill_system = (
        "You are a Tool-Making Agent.\n\n"
        "## CRITICAL RULES:\n"
        "1. ALWAYS start source_code with exactly this import:\n"
        "   from langchain_core.tools import tool\n"
        "   NEVER use agent_toolkit, langchain.tools, or any other import.\n\n"
        "2. Name the skill clearly in snake_case that describes what it does.\n"
        "   Good: calculate_bmi, convert_currency, fetch_stock_price\n"
        "   Bad: bmi_tool, tool1, my_calculator\n\n"
        "## STEPS:\n"
        "1. Understand what capability is needed.\n"
        "2. Write a Python @tool function as source_code.\n"
        "3. Write a short description.\n"
        "4. Write arguments as a valid JSON string for tool_schema_json.\n"
        "5. Call manage_skill to save it to the memory database.\n"
    )

    try:
        skill_agent = create_agent(model=llm, tools=[TOOLS_MANAGE])
        skill_result = skill_agent.invoke({"messages": [
            {"role": "system", "content": skill_system},
            {"role": "user",   "content": step.get("description", "") + prev_results_str},
        ]})

        # Search after creation → register into TOOL_REGISTRY
        registered_name = _search_and_register_skill(step.get("description", ""))
        if registered_name:
            print(f"=== Skill Maker === Registered '{registered_name}' into TOOL_REGISTRY")

        # Patch agent_specs: replace placeholder tool names with the registered one
        updated_specs = state.get("agent_specs", [])
        if registered_name:
            updated_specs = []
            for spec in state.get("agent_specs", []):
                spec = dict(spec)
                spec["tools_required"] = [
                    registered_name if (t not in TOOL_REGISTRY and t != "manage_skill") else t
                    for t in spec.get("tools_required", [])
                ]
                updated_specs.append(spec)
            print(f"=== Skill Maker === Patched agent_specs with '{registered_name}'")

        print(f"=== Skill Maker === Done → back to action_observer (step {state.get('current_step_idx', 0) + 1})")
        return {
            "agent_results": {agent_name: skill_result},
            "agent_specs": updated_specs,
            "current_step_idx": state.get("current_step_idx", 0) + 1,
            "pending_skill_step": {},
        }

    except Exception as e:
        print(f"=== Skill Maker === ERROR: {e}")
        return {
            "agent_results": {agent_name: f"ERROR: {e}"},
            "current_step_idx": state.get("current_step_idx", 0) + 1,
            "pending_skill_step": {},
        }




# Routing for observer chain
def route_after_plan_observer(state: main_state):
    if state.get("plan_ready", False):
        return "agent_observer"
    if state.get("replan_count", 0) >= 3:  # force through หลัง 3 รอบ
        return "agent_observer"
    return "planner"


def route_after_agent_observer(state: main_state):
    if state.get("agents_ready", False):
        return "action_observer"
    if state.get("replan_count", 0) >= 3:
        return "action_observer"
    return "planner"


def route_after_action_observer(state: main_state):
    # ถ้ายังมี pending skill step → ไปสร้างที่ skill_maker ก่อน
    if state.get("pending_skill_step"):
        return "skill_maker"
    results = state.get("agent_results", {})
    replan_count = state.get("replan_count", 0)
    # ถ้ามี agent ที่ error และยังไม่เกิน 3 รอบ → replan
    if replan_count < 3 and any(str(v).startswith("ERROR:") for v in results.values()):
        print("=== Action Observer === errors detected → replan")
        return "plan_observer"
    return "fusion"


# Fusion and presentation nodes
def fusion_node(state: main_state):
    # ใช้ full conversation เพื่อให้รู้โจทย์หลักและข้อมูลที่ clarify มาทั้งหมด
    full_conversation = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state["messages"]
    ])
    raw_data = state.get('agent_results', {})
    agent_specs = state.get('agent_specs', [])

    prompt = (
        "You are an expert analyst and synthesizer.\n"
        "Your MOST IMPORTANT rule: Answer the user's original question IMMEDIATELY in the very first sentence, "
        "then provide supporting details.\n\n"
        f"## CONVERSATION HISTORY (original request + any clarifications):\n{full_conversation}\n\n"
        f"## AGENT SPECS USED:\n{agent_specs}\n\n"
        f"## RAW DATA FROM AGENTS:\n{raw_data}\n\n"
        "## OUTPUT RULES:\n"
        "1. Start with a direct answer to the user's original question — no filler intros.\n"
        "2. Provide 2-3 bullet points explaining the key findings or reasoning.\n"
        "3. If data was fetched (stocks, weather, etc.), include a concise Data Snapshot.\n"
        "4. If no external data was needed (e.g. calculation tasks), skip the Data Snapshot.\n"
        "5. Keep the entire response under 200 words. Be clear and direct."
    )

    results = llm.invoke(prompt)
    return {"fusion_results": results}



# ── Graph assembly ────────────────────────────────────────────────────────────

workflow = StateGraph(main_state)
workflow.add_node("input_validator",  input_validator_node)
workflow.add_node("planner",          planner_node)
workflow.add_node("plan_observer",    plan_observer_node)
workflow.add_node("agent_observer",   agent_observer_node)
workflow.add_node("action_observer",  action_observer_node)
workflow.add_node("skill_maker",      skill_maker_node)
workflow.add_node("fusion",           fusion_node)

workflow.set_entry_point("input_validator")

workflow.add_conditional_edges(
    "input_validator",
    route_after_validation,
    {"input_validator": "input_validator", "planner": "planner"},
)

workflow.add_edge("planner", "plan_observer")

workflow.add_conditional_edges(
    "plan_observer",
    route_after_plan_observer,
    {"planner": "planner", "agent_observer": "agent_observer"},
)

workflow.add_conditional_edges(
    "agent_observer",
    route_after_agent_observer,
    {"planner": "planner", "action_observer": "action_observer"},
)

workflow.add_conditional_edges(
    "action_observer",
    route_after_action_observer,
    {"skill_maker": "skill_maker", "plan_observer": "plan_observer", "fusion": "fusion"},
)

workflow.add_edge("skill_maker", "action_observer")
workflow.add_edge("fusion", END)

app = workflow.compile(checkpointer=MemorySaver())