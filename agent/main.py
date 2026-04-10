from graph import app
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    topic = input("Enter a topic: ")
    stream_mode = input("Enter stream_mode (comma separated, default: updates): ")
    stream_mode = ["updates"]

    for chunk in app.stream(
        {"messages": [HumanMessage(content=topic)]},
        stream_mode=stream_mode,
        version="v2",
    ):
        if chunk.get("type") == "updates":
            for node_name, state in chunk["data"].items():
                print(f"Node {node_name} updated: {state}")
