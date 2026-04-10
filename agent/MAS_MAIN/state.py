from typing import TypedDict, Annotated, Any
import operator
from langchain_core.messages import AnyMessage

class main_state(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    agent_tasks: dict
    agent_results: dict
    fusion_results: str
    # 🌟 ตัวแปรใหม่สำหรับจำโหมดการวิ่ง
    route_mode: str