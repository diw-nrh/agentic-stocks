import json
import requests
from langchain_core.tools import tool

@tool
def manage_skill(name: str, description: str, source_code: str, tool_schema_json: str) -> str:
    """
    Use this tool to CREATE or UPDATE a skill (tool) for the agent.
    - name: Unique skill name.
    - description: What the skill does.
    - source_code: The raw python source code containing the @tool function.
    - tool_schema_json: JSON string of the tool's schema arguments.
    """
    try:
        schema_dict = json.loads(tool_schema_json)
    except Exception as e:
        return f"Error parsing schema JSON: {e}"

    # ยิง Request ไปที่ Database ของ Memory Service
    url = "http://localhost:9004/skills/save"
    payload = {
        "name": name,
        "description": description,
        "source_code": source_code,
        "tool_schema": schema_dict
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return f"Success! Skill '{name}' saved to Memory Database."
    except Exception as e:
        return f"Error connecting to memory service: {e}"

TOOLS_MANAGE = manage_skill
