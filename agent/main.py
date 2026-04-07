from agent.graph import app
from langchain_core.messages import HumanMessage

while True:
    user_input = input(">> ")
    result = app.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    for message in result["messages"]:
        print(f"[{type(message).__name__}]: {message.content if message.content else message.tool_calls}")