from pydantic import BaseModel, Field
from typing import List, Dict

class PlanOption(BaseModel):
    option_id: int
    tasks: Dict[str, str] = Field(description="ใช้ key เป็นชื่อ agent (weather_agent, news_agent, stocks_agent) และ value เป็นคำสั่ง (กฏเหล็ก : ถ้าไม่ต้องใช้ agent ตัวใด ไม่จำเป็นต้องมี key นั้น)")
    rationale: str = Field(description="เหตุผลว่าทำไมแผนนี้ถึงดี")

class ToTGenerator(BaseModel):
    options: List[PlanOption] = Field(description="สร้างทางเลือกแผนงานมา 3 แบบที่แตกต่างกัน")

class ToTEvaluator(BaseModel):
    best_option_id: int = Field(description="ID ของแผนที่เหมาะสมที่สุด")
    critique: str = Field(description="เหตุผลที่เลือกแผนนี้และข้อควรระวัง")
    
