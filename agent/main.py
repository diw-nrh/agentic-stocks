from agent.graph import app
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    user_input = input(">> ")
    result = app.invoke({
        "messages": [HumanMessage(content=user_input)]})