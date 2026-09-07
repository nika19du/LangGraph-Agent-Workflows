from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from config import embeddings


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


vectorstore = InMemoryVectorStore(embeddings)

vectorstore.add_documents(
    [
        Document(page_content=text)
        for text in KNOWLEDGE
    ]
)