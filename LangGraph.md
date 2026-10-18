# Topic 6: Persistent Conversation Memory with LangGraph and SQLite

## Introduction

In the previous topic, I learned how to maintain conversation history using in-memory message history. However, that approach has one major limitation: the conversation history is lost when the Python program stops.

To solve this problem, I explored persistent conversation memory using **LangGraph and SQLite**.

The idea is simple: instead of keeping conversation history only in the program's memory, we save the graph's state in a database. When the application runs again, it can retrieve the previous conversation and continue from where it left off.

## 1. Why do we need persistent memory?

Consider a chatbot that remembers something I told it during a conversation. If I close the program and run it again, a normal Python dictionary or an in-memory history object will no longer contain that conversation.

Persistent memory solves this problem by storing the conversation state outside the running Python process.

For this implementation, I used SQLite, which stores data in a local database file.

One important thing to understand is that Gemini does not automatically remember previous conversations. LangGraph retrieves the saved messages and passes them to the model as context.

## 2. Important concepts

Before implementing persistent memory, I learned the following concepts.

- **State:** The data maintained and updated while a graph executes. In this project, the state contains the conversation messages.
- **Node:** A function that performs a particular task. My chatbot node receives the conversation messages, invokes Gemini, and returns the response.
- **Graph:** Defines the workflow by connecting nodes and specifying the execution order.
- **Checkpointer:** Saves and restores the graph's state.
- **SQLite:** A lightweight database that stores information in a local file.
- **Thread ID:** Identifies a particular conversation so that its saved state can be retrieved later.
- **MessagesState:** A LangGraph state schema designed to manage conversation messages.

The most important relationship to remember is:

**Same database + same thread ID = access to the same conversation history.**

Using a different thread ID creates a separate conversation thread.

## 3. Setting up the model

I used `ChatGoogleGenerativeAI` to interact with Gemini. Since the application works with chat messages, a chat-model interface is appropriate here.

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

secret_key = os.getenv("SECRET_KEY_GEMINI")

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.5,
    google_api_key=secret_key
)
```

I used `python-dotenv` to load the API key from my `.env` file instead of hardcoding it in the source code.

I also make sure not to push the `.env` file to GitHub, since it contains a secret API key.

## 4. Creating the chatbot node

A node is simply a Python function that performs a task in the graph.

My chatbot node takes the current state, extracts the messages, sends them to Gemini, and returns the response.

```python
def chatbot(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
```

Here is what happens inside the function:

1. `state["messages"]` retrieves the conversation messages available to the node.
2. `model.invoke()` sends those messages to Gemini.
3. The model returns an AI message.
4. The function returns that message under the `"messages"` key.

I return `[response]` as a list because `MessagesState` uses a message reducer that appends returned messages to the existing conversation state.

## 5. Building the graph

After defining the chatbot node, I created a graph using `StateGraph`.

```python
from langgraph.graph import MessagesState, START, END, StateGraph

builder = StateGraph(MessagesState)

builder.add_node("Chatbot", chatbot)

builder.add_edge(START, "Chatbot")
builder.add_edge("Chatbot", END)
```

The graph has a simple execution flow:

```text
START
  |
  v
Chatbot Node
  |
  v
END
```

`StateGraph(MessagesState)` creates the graph builder using a messages-based state schema.

`add_node("Chatbot", chatbot)` registers the chatbot function under the name `"Chatbot"`.

`add_edge(START, "Chatbot")` tells the graph to execute the chatbot node at the beginning.

`add_edge("Chatbot", END)` tells the graph to finish after the chatbot node completes.

At this point, I have defined the workflow, but I still need to configure persistence.

## 6. Adding SQLite persistence

I used `SqliteSaver` from `langgraph-checkpoint-sqlite` to save the graph's state in a SQLite database.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("chat_history.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

The database file is named `chat_history.db`.

`SqliteSaver.from_conn_string()` creates the SQLite-backed checkpointer, and the `with` block manages its lifetime.

The important line is:

```python
graph = builder.compile(checkpointer=checkpointer)
```

This attaches the checkpointer to the graph. Without it, the graph would not use this checkpointer to save and restore its state.

I do not need to create another SQLite connection separately for this implementation because the saver manages its own connection.

## 7. Using a thread ID

To tell LangGraph which conversation to load and save, I pass a thread ID through the graph's configuration.

```python
config = {
    "configurable": {
        "thread_id": "vimanthan_chat"
    }
}
```

The thread ID is `"vimanthan_chat"` in this example.

When I invoke the graph, LangGraph uses this ID to identify the conversation associated with the saved state.

```python
result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Remember that my favourite programming language is Python."
            }
        ]
    },
    config=config
)

print(result["messages"][-1].content)
```

The input contains a new user message. The graph processes it, returns the AI response, and saves the updated state through the checkpointer.

`result["messages"][-1].content` accesses the content of the final message in the returned state.

## 8. Testing persistence across program restarts

To verify that persistent memory works, I tested the conversation across separate program runs.

**First run:**

I sent the message:

> Remember that my favourite programming language is Python.

The graph processed the message and saved the updated conversation state to `chat_history.db`.

**Second run:**

After stopping and restarting the program, I used the same database file and the same thread ID. This time, I sent a new question:

> What is my favourite programming language?

LangGraph restored the previous conversation state for that thread. The model could then use the earlier message as context when answering the new question.

There is one important detail: I must send the new question on the second run. If I execute the original input again, I am submitting the original statement again.

This test helped me understand how persistent conversation memory works in practice.

## 9. Key takeaways

- In-memory history is temporary and does not survive a process restart.
- SQLite provides persistent storage for the graph's saved state.
- A checkpointer saves and restores graph state.
- `MessagesState` manages the conversation messages.
- A node performs a task, while a graph defines the workflow.
- `builder.compile(checkpointer=checkpointer)` enables the graph to use the configured checkpointer.
- The thread ID identifies which conversation should be resumed.
- The same database file and thread ID allow a conversation to be continued across program restarts.
- Persistent storage does not give the model permanent memory by itself. The application restores the messages and supplies them as context.

## Conclusion

In this topic, I learned how to move from temporary in-memory conversation history to persistent conversation memory using LangGraph and SQLite.

I now understand how state, nodes, graphs, checkpointers, and thread IDs work together to preserve conversation history. This provides a foundation for building chatbots that can resume conversations even after the application has been restarted.
