import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (ChatPromptTemplate,MessagesPlaceholder)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()
google_gemini_key=os.getenv("SECRET_KEY_GEMINI")
store={}
def get_session_his(session_id : str):
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]

model=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temperature=0.5,google_api_key=google_gemini_key)

history=InMemoryChatMessageHistory()


prompt=ChatPromptTemplate.from_messages([
    ("system","You are an AI"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}")
])

pipe=prompt | model |StrOutputParser()

""" response=pipe.invoke({
    "history":history.messages,
    "question":"Tell me what you remember from this chat"
})

print(response)

 """

store["user_1"]=history

chat_message_history=RunnableWithMessageHistory(
    pipe,
    get_session_his,
    input_messages_key="question",
    history_messages_key="history"
)

config={
    "configurable":{
        "session_id":"user_1",
    }
}


response1 = chat_message_history.invoke(
    {"question": "My favourite language is Python."},
    config=config
)
print(response1)

response2 = chat_message_history.invoke(
    {"question": "What is my favourite language?"},
    config=config
)
print(response2)

print(store["user_1"].messages.HumanMessages())