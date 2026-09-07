from config import llm
from rag.knowledge import vectorstore
from state import State


def prompt_llm_rag(state: State):
    query = state["messages"][-1].content

    documents = vectorstore.similarity_search(
        query,
        k=3
    )

    context = "\n".join(
        f"- {doc.page_content}"
        for doc in documents
    )

    messages = [
        {
            "role": "system",
            'content': f'You are a RAG agent. Answer the user using only the context below. If the answer is not in it, say you don\'t know.\n\nContext:\n{context}',
        }
    ] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [
            {
                "role": "assistant",
                "content": response.content
            }
        ]
    }