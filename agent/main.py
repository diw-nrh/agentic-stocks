from graph import app
from langchain_core.messages import HumanMessage
from langgraph.types import Command


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

    if "agent_results" in state:
        readable = {}
        for key, value in state["agent_results"].items():
            readable[key] = extract_agent_result(value)
        return readable

    if "fusion_results" in state:
        return extract_content(state["fusion_results"])


    return {k: extract_content(v) for k, v in state.items() if k != "messages"}


def run_stream(input_data, config):
    """รัน graph และ handle interrupt (Human-in-the-Loop) อัตโนมัติ"""
    while True:
        interrupted = False
        for chunk in app.stream(input_data, config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupt_info = chunk["__interrupt__"]
                question = interrupt_info[0].value.get("question", "Please provide more information: ")
                print(f"\n[Agent]: {question}")
                user_answer = input("You: ").strip()
                input_data = Command(resume=user_answer)
                interrupted = True
                break

            for node_name, state in chunk.items():
                if node_name == "__interrupt__":
                    continue
                formatted = format_update(node_name, state)
                if formatted:
                    print(f"\n[{node_name}]: {formatted}")

        if not interrupted:
            break


if __name__ == "__main__":
    topic = input("Enter a topic: ").strip()
    config = {"configurable": {"thread_id": "main-session"}}
    run_stream({"messages": [HumanMessage(content=topic)]}, config)