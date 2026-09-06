import os
import uuid
import subprocess
from typing import TypedDict, Annotated, Literal

from dotenv import load_dotenv
from langchain_core import documents
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = init_chat_model('openai:gpt-4.1-mini')

KNOWLEDGE = [
    "LangGraph is a library for building stateful, multi-agent applications on top of LangChain.",
    "A StateGraph in LangGraph defines nodes and edges that operate on a shared typed state.",
    "Checkpointers like InMemorySaver let LangGraph persist conversation state across invocations using a thread_id.",
    "Reducers define how multiple updates to the same state field are combined. For example, add_messages appends and manages chat messages in the state.",
    "Conditional edges allow LangGraph to route execution dynamically based on the current state.",
    "START and END are special reserved nodes in LangGraph. START represents the entry point of the graph and END represents termination.",
    "RAG, or Retrieval-Augmented Generation, combines a retriever over a knowledge base with an LLM to ground answers in retrieved documents.",
    "A vector store stores document embeddings and supports similarity search for relevant documents.",
    "Embeddings convert text into numerical vectors so semantically similar texts can be found using vector similarity.",
    "InMemoryVectorStore is an in-memory vector database provided by LangChain. It is useful for demos and experiments but is not intended for durable production storage.",
    "LangSmith is a platform for tracing, debugging, evaluating, and monitoring LLM and agent applications.",
    "LangSmith traces record the execution of chains, tools, models, and agents, making it easier to inspect what happened during a run.",
    "LangSmith datasets can be used to store test examples and evaluate the behavior of LLM applications across multiple inputs.",
    "LangSmith evaluations can use deterministic checks, custom evaluators, or LLM-as-a-judge approaches to measure application quality.",
    "Human-in-the-loop workflows allow a LangGraph execution to pause before sensitive actions and resume after a user approves or rejects the action.",
    "A LangGraph Store is used for long-term memory that can be shared across different conversation threads.",
    "A checkpointer is primarily used for thread-level state persistence, while a Store can hold information across different threads."
]

vectorstore = InMemoryVectorStore(OpenAIEmbeddings(model = 'text-embedding-3-small'))
vectorstore.add_documents([Document(page_content=text) for text in KNOWLEDGE])

class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description = 'Classify whether the user wants to just chat, ask for knowledge or change code in the project.')


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None # passing node to node

def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {
            'role': 'system',
            'content': 'Determine / classify whether the user wants to chat ("chat"), retrieve knowledge ("knowledge") or change code ("code).',
        },
        {
            'role': 'user',
            'content': state['messages'][-1].content
        }
    ])

    return  {'message_intent': result.message_intent}

def prompt_llm_chat(state: State):
    messages = [
        {'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice.'},
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}

def prompt_llm_rag(state: State):
    query = state['messages'][-1].content

    documents = vectorstore.similarity_search(query, k = 3)

    context = '\n'.join(f'- {doc.page_content}' for doc in documents)

    messages = [
        {
            'role': 'system',
            'content': f'You are a RAG agent. Answer the user using only the context below. If the answer is not in it, say you don\'t know.\n\nContext:\n{context}',
        }
    ]+ state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}


def prompt_llm_code(state: State):
    user_prompt = state['messages'][-1].content

    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')

    result = subprocess.run(
        ['claude', '-p', user_prompt, '--permission-mode', 'acceptEdits'],
        cwd = workspace,
        capture_output= True,
        text = True
    )

    output = result.stdout.strip() or result.stderr.strip()

    return {'messages': [{'role': 'assistant', 'content': output}]}


graph_builder = StateGraph(State)

graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)

graph_builder.add_edge(START, 'classifier')
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {'chat': 'chat_agent', 'knowledge':'rag_agent', 'code': 'coding_agent'})

graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {'configurable':{'thread_id':uuid.uuid4()}}

while True:
    user_message = input('Enter a message: ')
    result = graph.invoke(
        {
            'messages':[{'role':'user', 'content': user_message}],
        },
        config = config
    )

    print(result['messages'][-1].content)





