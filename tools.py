
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import ToolMessage

load_dotenv()
google_gemini_key = os.getenv("SECRET_KEY_GEMINI")

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.5,
    google_api_key=google_gemini_key
)

@tool
def add(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b

model_with_tools = model.bind_tools([add])

response = model_with_tools.invoke(
    "Use the add tool to calculate 15 + 25. You must use the add tool."
)


from langchain_core.messages import HumanMessage, ToolMessage

user_message = HumanMessage(
    content="Use the add tool to calculate 15 + 25."
)

response = model_with_tools.invoke([user_message])

if not response.tool_calls:
    print("No tool call:", response.content)
else:
    tool_call = response.tool_calls[0]
    result = add.invoke(tool_call["args"])

    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],
        name=tool_call["name"]
    )

    final_response = model_with_tools.invoke([
        user_message,
        response,
        tool_msg
    ])

    print(final_response.content)
