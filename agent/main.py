from graph import app
from langchain_core.messages import HumanMessage


def extract_content(val):
    if isinstance(val, dict) and "content" in val:
        return val["content"]
    if isinstance(val, dict) and "messages" in val:
        return "".join(getattr(m, "content", str(m)) for m in val["messages"])
    if isinstance(val, list):
        return "".join(getattr(m, "content", str(m)) for m in val)
    if hasattr(val, "content"):
        return val.content
    return str(val)


def extract_agent_result(result):
    if not isinstance(result, dict):
        return extract_content(result)

    messages = result.get("messages")
    if not isinstance(messages, list):
        return extract_content(result)

    tool_outputs = []
    ai_outputs = []
    for msg in messages:
        msg_type = getattr(msg, "type", "")
        content = getattr(msg, "content", "")
        if msg_type == "tool" and content:
            tool_outputs.append(content)
        elif msg_type == "ai" and content:
            ai_outputs.append(content)

    if tool_outputs:
        unique_outputs = []
        seen = set()
        for output in tool_outputs:
            if output not in seen:
                seen.add(output)
                unique_outputs.append(output)
        return "\n".join(unique_outputs)
    if ai_outputs:
        return ai_outputs[-1]
    return ""


def format_update(node_name, state):
    if not isinstance(state, dict):
        return extract_content(state)

    if "agent_tasks" in state:
        return state["agent_tasks"]

    if "agent_results" in state:
        readable = {}
        for key, value in state["agent_results"].items():
            readable[key] = extract_agent_result(value)
        return readable

    if "fusion_results" in state:
        return extract_content(state["fusion_results"])


    return {k: extract_content(v) for k, v in state.items()}

if __name__ == "__main__":
    topic = input("Enter a topic: ")
    stream_mode = input("Enter stream_mode (comma separated, default: updates): ")
    if stream_mode.strip():
        stream_mode = [m.strip() for m in stream_mode.split(",") if m.strip()]
    else:
        stream_mode = ["updates"]

    for chunk in app.stream(
        {"messages": [HumanMessage(content=topic)]},
        stream_mode=stream_mode,
        version="v2",
    ):
        if chunk.get("type") == "updates":
            for node_name, state in chunk["data"].items():
                print(f"Node {node_name} updated: {format_update(node_name, state)}")
        elif chunk.get("type") == "custom":
            print(f"Status: {extract_content(chunk['data'])}")