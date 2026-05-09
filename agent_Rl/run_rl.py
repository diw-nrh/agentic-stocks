"""
compare_agent.py — LangGraph Inference + Token Tracking
เปรียบเทียบ Base Model vs RL Model ในแง่:
  - จำนวน token ที่ใช้
  - ความสำเร็จในการเรียก tool
  - คุณภาพ JSON output
"""
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from tools.registry import TOOLS_NEWS, TOOLS_STOCK, TOOLS_WEATHER

# ──────────────────────────────────────────────
# 1. Config & Tool Registry
# ──────────────────────────────────────────────
BASE_MODEL_ID  = "Qwen/Qwen2.5-0.5B-Instruct"
RL_MODEL_PATH  = "./grpo_planner_model_final"

TOOL_MAP: dict[str, Any] = {
    "weather": TOOLS_WEATHER,
    "stock":   TOOLS_STOCK,
    "news":    TOOLS_NEWS,
}

SYSTEM_PROMPT = (
    "You are an AI planner. Respond ONLY with a valid JSON object in this exact format:\n"
    '{"tool_name": "<tool>", "arguments": { ... }}\n'
    "Available tools: weather, stock, news\n"
    "Do not add any explanation. Output JSON only."
)


# ──────────────────────────────────────────────
# 2. Token Stats (dataclass เก็บผล per run)
# ──────────────────────────────────────────────
@dataclass
class TokenStats:
    prompt_tokens:     int = 0
    completion_tokens: int = 0
    total_tokens:      int = 0
    tool_success:      bool = False
    tool_result:       str = ""
    raw_response:      str = ""
    latency_sec:       float = 0.0
    error:             str = ""

    @property
    def summary(self) -> str:
        status = "✅" if self.tool_success else "❌"
        return (
            f"{status} tokens(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens}) | "
            f"latency={self.latency_sec:.2f}s"
        )


# ──────────────────────────────────────────────
# 3. Model Wrapper (load once, reuse)
# ──────────────────────────────────────────────
class LocalLLM:
    """โหลด HuggingFace model ครั้งเดียวและนับ token ด้วย tokenizer ตัวเอง"""

    def __init__(self, model_id: str, label: str = ""):
        self.label = label or model_id
        print(f"[{self.label}] กำลังโหลด model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        print(f"[{self.label}] โหลดเสร็จ ✓")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, prompt: str, max_new_tokens: int = 128) -> tuple[str, TokenStats]:
        """Generate text และคืน (response_text, TokenStats)"""
        prompt_tokens = self.count_tokens(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy — reproducible
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0

        # ตัด prompt ออก เหลือแค่ส่วนที่โมเดลสร้างใหม่
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        completion_tokens = self.count_tokens(response)

        stats = TokenStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            raw_response=response,
            latency_sec=latency,
        )
        return response, stats


# ──────────────────────────────────────────────
# 4. LangGraph State & Nodes
# ──────────────────────────────────────────────
class AgentState(TypedDict):
    query:      str
    prompt:     str
    response:   str
    tool_call:  dict
    stats:      TokenStats
    llm:        LocalLLM      # ส่ง llm instance ผ่าน state


def build_prompt(query: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


# Node 1: LLM plan → สร้าง tool call JSON
def plan_node(state: AgentState) -> AgentState:
    llm: LocalLLM = state["llm"]
    prompt = build_prompt(state["query"])
    response, stats = llm.generate(prompt)
    state["prompt"]   = prompt
    state["response"] = response
    state["stats"]    = stats
    return state


# Node 2: Execute tool จาก JSON ที่โมเดลสร้าง
def execute_node(state: AgentState) -> AgentState:
    stats: TokenStats = state["stats"]
    obj = _extract_json(state["response"])

    if obj is None:
        stats.error = "JSON parse failed"
        state["tool_call"] = {}
        return state

    state["tool_call"] = obj
    tool_name = obj.get("tool_name", "")
    arguments = obj.get("arguments", {}) or {}

    tool = TOOL_MAP.get(tool_name)
    if tool is None:
        stats.error = f"Unknown tool: {tool_name}"
        stats.tool_success = False
        return state

    try:
        result = tool.invoke(arguments)
        stats.tool_success = True
        stats.tool_result  = str(result)
    except Exception as exc:
        stats.tool_success = False
        stats.error = str(exc)

    return state


def build_agent_graph() -> Any:
    """สร้าง LangGraph สำหรับ inference"""
    g = StateGraph(AgentState)
    g.add_node("plan",    plan_node)
    g.add_node("execute", execute_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)
    return g.compile()


AGENT = build_agent_graph()


def run_agent(query: str, llm: LocalLLM) -> TokenStats:
    """รัน agent 1 query แล้วคืน TokenStats"""
    final_state = AGENT.invoke({
        "query":     query,
        "prompt":    "",
        "response":  "",
        "tool_call": {},
        "stats":     TokenStats(),
        "llm":       llm,
    })
    return final_state["stats"]


# ──────────────────────────────────────────────
# 5. Comparison Runner
# ──────────────────────────────────────────────
@dataclass
class ModelReport:
    label:         str
    stats_list:    list[TokenStats] = field(default_factory=list)

    @property
    def total_queries(self) -> int:
        return len(self.stats_list)

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.stats_list if s.tool_success)

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_queries if self.total_queries else 0.0

    @property
    def avg_prompt_tokens(self) -> float:
        return sum(s.prompt_tokens for s in self.stats_list) / max(self.total_queries, 1)

    @property
    def avg_completion_tokens(self) -> float:
        return sum(s.completion_tokens for s in self.stats_list) / max(self.total_queries, 1)

    @property
    def avg_total_tokens(self) -> float:
        return sum(s.total_tokens for s in self.stats_list) / max(self.total_queries, 1)

    @property
    def avg_latency(self) -> float:
        return sum(s.latency_sec for s in self.stats_list) / max(self.total_queries, 1)

    def print_summary(self):
        print(f"\n{'='*55}")
        print(f"  📊 {self.label}")
        print(f"{'='*55}")
        print(f"  Queries        : {self.total_queries}")
        print(f"  Success rate   : {self.success_rate*100:.1f}%  ({self.success_count}/{self.total_queries})")
        print(f"  Avg tokens     : {self.avg_total_tokens:.1f}  "
              f"(prompt={self.avg_prompt_tokens:.1f}, completion={self.avg_completion_tokens:.1f})")
        print(f"  Avg latency    : {self.avg_latency:.2f}s")
        print(f"{'='*55}")


def compare(queries: list[str]):
    """โหลดทั้ง 2 โมเดล แล้วรันทุก query เปรียบเทียบกัน"""

    # โหลด 2 โมเดล
    base_llm = LocalLLM(BASE_MODEL_ID,  label="Base (No RL)")
    rl_llm   = LocalLLM(RL_MODEL_PATH,  label="RL (GRPO)")

    base_report = ModelReport(label="Base Model (No RL)")
    rl_report   = ModelReport(label="RL Model  (GRPO)")

    print(f"\n🚀 รัน {len(queries)} queries บนทั้ง 2 โมเดล...\n")

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] query: {query[:50]}...")

        # Base model
        base_stats = run_agent(query, base_llm)
        base_report.stats_list.append(base_stats)
        print(f"  Base → {base_stats.summary}")
        if base_stats.raw_response:
            print(f"         response: {base_stats.raw_response[:80]}")

        # RL model
        rl_stats = run_agent(query, rl_llm)
        rl_report.stats_list.append(rl_stats)
        print(f"  RL   → {rl_stats.summary}")
        if rl_stats.raw_response:
            print(f"         response: {rl_stats.raw_response[:80]}")

        print()

    # Summary
    base_report.print_summary()
    rl_report.print_summary()

    # Delta
    print(f"\n{'='*55}")
    print("  📈 Delta (RL − Base)")
    print(f"{'='*55}")
    token_delta   = rl_report.avg_total_tokens - base_report.avg_total_tokens
    success_delta = (rl_report.success_rate - base_report.success_rate) * 100
    latency_delta = rl_report.avg_latency - base_report.avg_latency
    direction     = lambda v: ("+" if v >= 0 else "") + f"{v:.2f}"

    print(f"  Avg total tokens : {direction(token_delta)}")
    print(f"  Success rate     : {direction(success_delta)}%")
    print(f"  Avg latency      : {direction(latency_delta)}s")
    print(f"{'='*55}\n")

    # Per-query detail table
    print("  Per-query token breakdown:")
    print(f"  {'#':>3} | {'Query':40} | {'Base tok':>8} | {'RL tok':>6} | {'Δ tok':>6} | Base✓ | RL✓")
    print(f"  {'-'*3}-+-{'-'*40}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-------+----")
    for i, (q, bs, rs) in enumerate(
        zip(queries, base_report.stats_list, rl_report.stats_list), 1
    ):
        delta = rs.total_tokens - bs.total_tokens
        b_ok  = "✅" if bs.tool_success else "❌"
        r_ok  = "✅" if rs.tool_success else "❌"
        print(
            f"  {i:>3} | {q[:40]:40} | {bs.total_tokens:>8} | "
            f"{rs.total_tokens:>6} | {delta:>+6} | {b_ok:^5} | {r_ok:^4}"
        )

    return base_report, rl_report


# ──────────────────────────────────────────────
# 6. Entrypoint
# ──────────────────────────────────────────────
TEST_QUERIES = [
    "สภาพอากาศที่กรุงเทพตอนนี้เป็นยังไง?",
    "ราคาหุ้น PTT วันนี้เท่าไหร่?",
    "ข่าวเศรษฐกิจไทยล่าสุดมีอะไรบ้าง?",
    "อุณหภูมิที่เชียงใหม่ตอนนี้?",
    "ราคาหุ้น KBANK ล่าสุด?",
    "ข่าวการเมืองไทยวันนี้?",
    "weather in Phuket?",
    "ADVANC stock price today?",
]

if __name__ == "__main__":
    compare(TEST_QUERIES)