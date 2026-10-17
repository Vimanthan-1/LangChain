import os
from langchain_google_genai  import GoogleGenerativeAI
from langchain_core.prompts import (ChatPromptTemplate,MessagesPlaceholder)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
                                        
from dotenv import load_dotenv
load_dotenv()

secret_key = os.getenv("SECRET_KEY_GEMINI")

model=GoogleGenerativeAI(model="gemini-3.1-flash-lite",
                        temperature=0.3,
                        google_api_key=secret_key)

#Baics of LangChain
'''
response=model.invoke("Explain LangChain in 3 simple words")
print(response)'''


#ChatTemplate
'''prompt=ChatPromptTemplate.from_template(
    "Explain the {topic} to an intermediate python developer"
    "Using one practical example."
)'''

'''pipe=prompt | model

response=pipe.invoke({
    "topic":"recursion"
})


print(response)'''

#Output Parsers
'''parser=StrOutputParser()
pipe=prompt|model|parser'''
'''response=pipe.invoke({
    "topic":"binary search"
})

print(type(response))
print(response)'''

#inmemorychathistory
history=InMemoryChatMessageHistory()
history.add_user_message("I am learning LangChain")
history.add_user_message("You are helping me do it")

try:
    prompt=ChatPromptTemplate.from_messages([
        ("system","Act as a helpful python tutor"),
        MessagesPlaceholder(variable_name="history"),
        ("human","{question}")    
    ])
    
    pareser=StrOutputParser()

    pipe=prompt | model | pareser

    response=pipe.invoke({
        "history":history.messages,
        "question":"tell me about my history"
    })

    print(response)
except Exception as e:
    print(f"An error occured : {e}")