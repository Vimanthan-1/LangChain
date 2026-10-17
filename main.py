import os
from langchain_google_genai  import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

secret_key = os.getenv("SECRET_KEY_GEMINI")

model=GoogleGenerativeAI(model="gemini-3.8-flash",
                        temperature=0.3,max_tokens=300,
                        google_api_key=secret_key)
response=model.invoke("Explain LangChain in 3 simple words")
print(response)
