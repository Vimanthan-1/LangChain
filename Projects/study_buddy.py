import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

"""
This is a Program which is used for Demonstrating the persistent memory of an LLM using the LangGraph module and the Sqlite module. We have implemented a study buddy
for learning leetcode problems and can be viewed as shown below . The chat histroy is stored in the file names as 'Chat_history.db' in the directory Projects . We have successfully implemented the 
program and it has executed successfully 

"""

load_dotenv(dotenv_path="D:/Lang Chain/.env")
secret_key = os.getenv("SECRET_KEY_GEMINI")
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0.5, google_api_key=secret_key
)


class StudyBuddyState(MessagesState):
    topic: str


def study_buddy(state: StudyBuddyState):
    response = pipe.invoke({"topic": state["topic"], "messages": state["messages"]})
    return {"messages": [AIMessage(content=response)]}


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a tutor teaching a {topic} to an intermediate python developer who has some knowledge about the basics of python . Your job is to make hi =m understand the {topic} and make him a highly skilled professional on that topic . Do maintain professionalism during the whole conversation and be restricted from the use of potential bad words or curse words",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

pipe = prompt | model | StrOutputParser()

builder = StateGraph(StudyBuddyState)
builder.add_node("study_buddy", study_buddy)

builder.add_edge(START, "study_buddy")
builder.add_edge("study_buddy", END)

with SqliteSaver.from_conn_string("chat_history.db") as chat:
    graph = builder.compile(checkpointer=chat)
    config = {"configurable": {"thread_id": "Student_1"}}
    response = graph.invoke(
        {
            "topic": "Dijkstras",  # Change to your wish
            "messages": [{"role": "user", "content": "Teach me the solution"}],
        },
        config=config,
    )

    print(response["messages"][-1].content)
