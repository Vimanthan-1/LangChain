import sqlite3
import os
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState
from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import START, END, StateGraph

load_dotenv()
secret_key = os.getenv("SECRET_KEY_GEMINI")
model = GoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0.5, google_api_key=secret_key
)
conn = sqlite3.connect("chat_history.db")


# defining a state
def chatbot(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


# Builiding a Graph
builder = StateGraph(MessagesState)
builder.add_node("Chatbot", chatbot)  # Added a node

# Execution Flow
builder.add_edge(START, "Chatbot")
builder.add_edge("Chatbot", END)

with SqliteSaver.from_conn_string("chat_history.db") as checkpointer:
    print("Connected Successfully")
    graph = builder.compile(checkpointer=checkpointer)
    print("Executed Successfully")
    config = {"configurable": {"thread_id": "vimanthan_chat"}}
    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What is my favoutite programming language?",
                }
            ]
        },
        config=config,
    )

    print(result["messages"][-1].content)
