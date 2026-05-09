"""
train.py — GRPO Agentic RL (Fixed Version)
แก้ไข:
  - tokenizer ใช้ Qwen แทน tiktoken/gpt-4o-mini
  - ลบ LangGraph ออกจาก reward (ไม่จำเป็น ใช้ run_tool_call ตรง ๆ)
  - เพิ่ม schema_reward_func (partial credit)
  - ปรับ negative reward จาก -0.5 → 0.0
  - ขยาย dataset จาก 3 → 30 ตัวอย่าง
  - เพิ่ม train/eval split
  - เพิ่ม checkpoint saving
"""
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
from typing import Any
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

from tools.registry import TOOLS_NEWS, TOOLS_STOCK, TOOLS_WEATHER

# ──────────────────────────────────────────────
# 1. Tool registry & Skills
# ──────────────────────────────────────────────
TOOL_MAP: dict[str, Any] = {
    "weather": TOOLS_WEATHER,
    "stock": TOOLS_STOCK,
    "news": TOOLS_NEWS,
}

# Tokenizer ที่ตรงกับโมเดล (แก้จาก tiktoken/gpt-4o-mini)
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def load_skills() -> dict[str, str]:
    """โหลด SKILL.md ไฟล์สำหรับแต่ละ tool"""
    skills = {}
    base_path = Path(__file__).parent.parent / "skills"
    
    for tool_name in TOOL_MAP.keys():
        skill_file = base_path / tool_name / "SKILL.md"
        if skill_file.exists():
            skills[tool_name] = skill_file.read_text(encoding="utf-8")
        else:
            print(f"Warning: SKILL.md not found for {tool_name}")
    
    return skills


SKILLS = load_skills()



def count_tokens(text: str) -> int:
    """นับ token ด้วย Qwen tokenizer จริง"""
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def run_tool_call(tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
    """เรียก tool จาก registry แล้วคืน (success, result)"""
    tool = TOOL_MAP.get(tool_name)
    if tool is None:
        return False, f"Unknown tool: {tool_name}"
    try:
        result = tool.invoke(arguments)
        return True, str(result)
    except Exception as exc:
        return False, str(exc)


# ──────────────────────────────────────────────
# 2. Reward Functions
# ──────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """ดึง JSON object แรกที่เจอในข้อความ"""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    +0.5  → ตอบเป็น JSON ที่ parse ได้
     0.0  → parse ไม่ได้เลย
    """
    rewards = []
    for resp in completions:
        obj = _extract_json(resp)
        rewards.append(0.5 if obj is not None else 0.0)
    return rewards


def schema_reward_func(completions, **kwargs) -> list[float]:
    """
    +0.25 → มี key 'tool_name'
    +0.25 → มี key 'arguments' และเป็น dict
    รวมสูงสุด 0.5
    """
    rewards = []
    for resp in completions:
        obj = _extract_json(resp)
        score = 0.0
        if obj:
            if "tool_name" in obj:
                score += 0.25
            if "arguments" in obj and isinstance(obj["arguments"], dict):
                score += 0.25
        rewards.append(score)
    return rewards


def execution_reward_func(completions, **kwargs) -> list[float]:
    """
    +1.0 → JSON ถูกต้องและเรียก tool สำเร็จ
     0.0 → อย่างอื่น (ไม่ลงโทษ เพื่อให้ gradient เสถียรช่วงแรก)
    """
    rewards = []
    for resp in completions:
        obj = _extract_json(resp)
        if obj is None:
            rewards.append(0.0)
            continue
        tool_name = obj.get("tool_name", "")
        arguments = obj.get("arguments", {}) or {}
        success, _ = run_tool_call(tool_name, arguments)
        rewards.append(1.0 if success else 0.0)
    return rewards


# ──────────────────────────────────────────────
# 3. Dataset - Realistic API queries
# ──────────────────────────────────────────────
RAW_QUERIES = [
    # Weather - current conditions
    "What is the weather like in Thailand right now?",
    "Get current weather for Bangkok",
    "Show me weather in Chiang Mai currently",
    "Current weather conditions in Phuket",
    "What is the current weather in Pattaya?",
    
    # Weather - forecast
    "Show me the weather forecast for Chiang Mai next week",
    "Get 7-day forecast for Thailand",
    "What is the weather forecast for Bangkok?",
    "Forecast weather in Udon Thani",
    "Get forecast for Songkhla",
    
    # Weather - historical
    "What was the weather in Phuket on 2024-03-15?",
    "Get historical weather data for Bangkok on 2024-03-10",
    "Show weather in Thailand for 2024-03-20",
    "Historical weather in Chiang Mai on 2024-03-25",
    "Get past weather data for Krabi on 2024-03-12",
    
    # Stock - current price
    "What is the current stock price for AAPL?",
    "Get current price for GOOGL",
    "Show current stock data for MSFT",
    "What is the stock price for TSLA right now?",
    "Get current price for META",
    
    # Stock - range data
    "What is the stock range data for AAPL between 2024-03-20 and 2024-03-24?",
    "Get stock data for GOOGL from 2024-03-15 to 2024-03-25",
    "Show stock history for MSFT between 2024-03-01 and 2024-03-31",
    "Get historical stock data for TSLA from 2024-02-01 to 2024-03-15",
    "Stock price range for META between 2024-03-10 and 2024-03-20",
    
    # Stock - scheduled/forecast
    "What is the scheduled stock data for AAPL up to 2024-03-24?",
    "Get scheduled price forecast for GOOGL until 2024-03-25",
    "Show upcoming stock predictions for MSFT",
    "Get scheduled data for TSLA until 2024-03-31",
    "Stock forecast for META until 2024-04-01",
    
    # News - current
    "What is the current news data for weather in Thailand?",
    "Get current news about stock market in USA",
    "Show me latest news about weather in america",
    "Get current news for stock in Thailand",
    "What is the current news about technology?",
    
    # News - range
    "What is the news range data for weather in thailand between 2024-03-20 and 2024-03-24?",
    "Get news about stock in america from 2024-03-15 to 2024-03-25",
    "Show news data for weather in songkhla between 2024-03-10 and 2024-03-20",
    "Get news about economy between 2024-02-01 and 2024-03-15",
    "News range for technology from 2024-03-01 to 2024-03-31",
    
    # News - scheduled
    "What is the scheduled news data for weather in thailand up to 2024-03-24?",
    "Get scheduled news for stock in america until 2024-03-25",
    "Show scheduled news data for weather in america",
    "Get upcoming news about economy until 2024-04-01",
    "Scheduled news for technology until 2024-03-31",
]

raw_data = {"query": RAW_QUERIES}


def make_system_prompt() -> str:
    available_tools = ", ".join(sorted(TOOL_MAP.keys()))
    skills_content = "\n\n".join([
        f"## {tool_name.upper()} Tool\n{SKILLS.get(tool_name, 'No skill definition found')}"
        for tool_name in sorted(TOOL_MAP.keys())
    ])
    
    return (
        "You are an AI planner. Respond ONLY with a valid JSON object in this exact format:\n"
        '{"tool_name": "<tool>", "arguments": { ... }}\n'
        f"Available tools: {available_tools}\n\n"
        "## Tool Usage Guidelines\n"
        f"{skills_content}\n\n"
        "Do not add any explanation. Output JSON only."
    )


def format_dataset(example: dict) -> dict:
    system_prompt = make_system_prompt()
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{example['query']}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return {"prompt": prompt}


# ──────────────────────────────────────────────
# 4. Main training
# ──────────────────────────────────────────────
def main():
    # สร้าง dataset พร้อม train/eval split
    full_dataset = Dataset.from_dict(raw_data).map(format_dataset)
    split = full_dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = split["train"]   # ~25 ตัวอย่าง
    eval_dataset  = split["test"]    # ~5 ตัวอย่าง

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    training_args = GRPOConfig(
        output_dir="./grpo_planner_checkpoints",
        learning_rate=1e-5,
        num_generations=8,             # เพิ่มจาก 4 → 8 เพื่อให้ GRPO มี variance
        generation_batch_size=8,       # ต้องหารด้วย num_generations ลงตัว
        temperature=0.8,               # สุ่ม output หลากหลาย ไม่ greedy
        max_completion_length=128,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=1,
        save_steps=10,
        save_total_limit=3,
        remove_unused_columns=False,   # สำคัญ: ต้องเก็บ expected_tool column ไว้
    )


    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        reward_funcs=[
            format_reward_func,      # +0.5  → JSON parseable
            schema_reward_func,      # +0.5  → schema ถูกต้อง
            execution_reward_func,   # +1.0  → tool run ผ่าน
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("เริ่ม GRPO training...")
    trainer.train()

    trainer.save_model("./grpo_planner_model_final")
    TOKENIZER.save_pretrained("./grpo_planner_model_final")
    print("บันทึกโมเดลที่ ./grpo_planner_model_final เสร็จสิ้น")


if __name__ == "__main__":
    main()