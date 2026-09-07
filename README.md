<img src="https://github.com/nika19du/LangGraph-Agent-Workflows/blob/master/graph.png?raw=true" alt="graph" />

# LangGraph Agent Workflow

A multi-route AI agent workflow built with LangGraph that classifies user requests
and routes them to different specialized nodes.

The application supports:

- General conversation
- Knowledge retrieval using RAG
- Coding tasks using Claude Code
- Human-in-the-Loop approval before code changes
- Conversation state using LangGraph checkpointing

## Workflow

Every user message first goes through an **Intent Classifier**.

The classifier analyzes the request and routes it into one of three categories:

1. **Chat**
2. **Knowledge Request**
3. **Coding Request**

### 1. Chat

General conversation is routed to the `chat_agent`.

The node uses an LLM with a conversational system prompt to generate a response.

Flow:

User → Classifier → Chat Agent → END

### 2. Knowledge Request

Questions related to the application's knowledge base are routed to the RAG agent.

The RAG workflow:

1. Converts the user query into an embedding.
2. Performs similarity search against the vector store.
3. Retrieves the most relevant documents.
4. Provides the retrieved context to the LLM.
5. Generates an answer grounded only in the retrieved knowledge.

Flow:

User → Classifier → RAG Agent → END

The current knowledge base contains information about LangGraph, LangSmith,
RAG, vector stores, embeddings, reducers, checkpointers, and related concepts.

### 3. Coding Request

Requests that require modifying project files are routed to the coding workflow.

Before Claude Code receives the request, an additional node reformulates the
user's request into a clear coding instruction using the conversation history.

Flow:

User
→ Classifier
→ Prepare Coding Request
→ Human Approval
→ Claude Code
→ END

Before Claude Code is allowed to modify the workspace, the workflow pauses
using LangGraph's Human-in-the-Loop functionality.

The user can:

- **Approve** → The request is sent to Claude Code.
- **Reject** → The coding workflow terminates without executing the request.
- **Revise** → The user provides a revised request, which is reformulated and
  sent through the approval process again.

This creates an approval loop:

Prepare Request
→ Human Approval
→ Revise
→ Prepare Request
→ Human Approval

Once approved, Claude Code executes the request inside the configured
workspace and the workflow terminates.

## Conversation State

The workflow uses LangGraph's `InMemorySaver` checkpointer with a `thread_id`
to preserve conversation state between invocations.

This allows follow-up requests to be interpreted using previous messages in
the same conversation.

For example:

User: My name is Nikol.
User: Add my name to README.md.

The classifier and prompt preparation nodes can use the conversation history
to understand what "my name" refers to.
