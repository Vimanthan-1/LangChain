# Learning LangChain with Gemini

These are my notes from learning LangChain using Google's Gemini API. I'm documenting each concept as I learn it, along with examples and mistakes I encountered along the way.

## What I've Learned So Far

- Connecting LangChain to Google's Gemini API
- Creating prompts using `ChatPromptTemplate`
- Connecting components using chains
- Processing model responses with output parsers
- Working with conversation history

---

## 1. Setting Up Gemini

I use `ChatGoogleGenerativeAI` to connect LangChain to Gemini. The API key is stored in a `.env` file instead of being written directly in the Python code.

### Install the required packages

```bash
uv add langchain langchain-google-genai python-dotenv
```

### Store the API key

Create a `.env` file in the project directory:

```env
GOOGLE_API_KEY=your_api_key_here
```

Load the environment variables in Python:

```python
from dotenv import load_dotenv

load_dotenv()
```

### Initialize the model

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
)
```

The model ID should be one that's available to my Google AI Studio account.

---

## 2. Prompt Templates

A prompt is the input or instruction given to a language model. Instead of writing a new prompt every time, I can create a reusable template with placeholders.

For this, LangChain provides `ChatPromptTemplate`.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} to an intermediate Python developer "
    "using one practical example."
)
```

Here, `{topic}` is a placeholder. I can supply a different topic whenever I use the prompt.

```python
formatted_prompt = prompt.invoke({
    "topic": "binary search"
})
```

**Key takeaway:** A prompt template prepares the input for the model. It doesn't call Gemini by itself.

---

## 3. Chains

A chain connects multiple LangChain components so that the output of one component becomes the input to the next.

LangChain uses the `|` operator to connect these components.

```python
chain = prompt | model

response = chain.invoke({
    "topic": "recursion"
})

print(response.content)
```

In this example:

1. The prompt template fills in the topic.
2. The formatted prompt is passed to Gemini.
3. Gemini generates a response.
4. The response is returned as an AI message.

I can use `response.content` to access the text in that message.

---

## 4. Output Parsers

The response from a model isn't always in the format I want for my Python program. Output parsers help convert model responses into more convenient representations.

### Using `StrOutputParser`

`StrOutputParser` extracts the text from an AI message and returns it as a regular Python string.

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({
    "topic": "binary search"
})

print(response)
print(type(response))
```

The response type should be:

```text
<class 'str'>
```

This also explains an error I encountered:

```text
AttributeError: 'str' object has no attribute 'content'
```

The error occurs when I try to access `.content` on a response that's already a string.

```python
# When the response is an AI message
print(response.content)

# When the response is a string
print(response)
```

**Key takeaway:** The output type depends on the components in the chain. `StrOutputParser` returns a string, making it easier to print or process the text.

---

## 5. Conversation History

A chatbot often needs information from earlier messages to answer follow-up questions correctly. For example, if I tell it that I'm learning LangChain and later ask what I'm learning, it needs access to that earlier message.

Conversation history lets an application store messages and provide them to the model when needed.

### Storing messages

LangChain provides `InMemoryChatMessageHistory` for storing conversation messages temporarily.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()

history.add_user_message("I am learning LangChain.")
history.add_ai_message("I am helping you learn it.")

print(history.messages)
```

I can use `add_user_message()` and `add_ai_message()` to store messages with the appropriate roles.

This history exists in memory, so it isn't preserved after the program exits.

### Adding history to a prompt

`MessagesPlaceholder` reserves a position in the prompt for previous messages.

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as a helpful Python tutor."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
```

Here's what each part does:

- `system` provides the model's instructions.
- `MessagesPlaceholder` creates a slot for conversation history.
- `human` represents the current user message.
- `{question}` is replaced with the question supplied when invoking the prompt.

The name `"history"` is the input key used for the placeholder. It isn't the history object itself.

### Passing the history to the model

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = prompt | model | parser

response = chain.invoke({
    "history": history.messages,
    "question": "What am I learning?"
})

print(response)
```

The dictionary supplies both the previous messages and the current question.

The history is passed using `history.messages`, which contains the stored messages. The `"history"` key matches the variable name in `MessagesPlaceholder`.

**One important distinction:** Storing messages doesn't automatically make them available to Gemini. My application needs to include the history in the input sent through the chain.

This example passes existing history to the model, but it doesn't automatically store the new question and answer after every request.

---

## 6. Putting It All Together

Here's a complete example combining the concepts covered so far.

```python
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
)

# Create conversation history
history = InMemoryChatMessageHistory()

history.add_user_message("I am learning LangChain.")
history.add_ai_message("I am helping you learn it.")

# Create the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as a helpful Python tutor."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# Create the chain
parser = StrOutputParser()
chain = prompt | model | parser

# Send the history and current question
response = chain.invoke({
    "history": history.messages,
    "question": "What am I learning?"
})

print(response)
```

The example assumes the dependencies are installed and the API key is configured correctly.

---

## 7. Key Takeaways

| Concept | What it does |
|---|---|
| `ChatGoogleGenerativeAI` | Connects LangChain to Gemini |
| `ChatPromptTemplate` | Creates reusable chat prompts |
| `invoke()` | Runs a component or chain with the supplied input |
| `\|` | Connects compatible components into a chain |
| `StrOutputParser` | Extracts the model's text as a Python string |
| `InMemoryChatMessageHistory` | Stores messages temporarily |
| `MessagesPlaceholder` | Inserts supplied conversation messages into a prompt |

### What I want to learn next

- Automatically saving new messages and responses
- Building a chatbot that maintains conversation history across turns
- Understanding more LangChain components and LCEL
- Working with structured outputs and JSON
- Connecting multiple chains for more complex tasks

This repository will grow as I continue learning and experimenting with LangChain.
