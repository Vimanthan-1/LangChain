import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools import tool

load_dotenv()
google_gemini_key = os.getenv("SECRET_KEY_GEMINI")
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0.5, google_api_key=google_gemini_key
)


@tool
def add(a: int, b: int) -> int:
    """This is a function to add"""
    return a + b


'''print(add.name)
print(add.description)
print(add.args)
ans = add.invoke({"a": 2, "b": 2})
print(ans)
'''

model_with_tools = model.bind_tools([add])
response = model_with_tools.invoke(
    "Use the add tool to calculate 2 + 2."
)


if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = add.invoke(tool_call["args"])
    print(result)

    # Your task: execute the requested tool here

