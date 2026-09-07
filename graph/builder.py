from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from state import State
from nodes.classifier import classify_intent
from nodes.chat import prompt_llm_chat
from nodes.rag import prompt_llm_rag
from nodes.coding import (
    prepare_coding_request,
    prompt_llm_code
)
from nodes.approval import accept_coding


def build_graph():
    builder = StateGraph(State)

    builder.add_node("classifier", classify_intent)
    builder.add_node("chat_agent", prompt_llm_chat)
    builder.add_node("rag_agent", prompt_llm_rag)
    builder.add_node(
        "prepare_coding_request",
        prepare_coding_request
    )
    builder.add_node(
        "accept_coding",
        accept_coding
    )
    builder.add_node(
        "coding_agent",
        prompt_llm_code
    )

    builder.add_edge(
        START,
        "classifier"
    )

    builder.add_conditional_edges(
        "classifier",
        lambda state: state["message_intent"],
        {
            "chat": "chat_agent",
            "knowledge": "rag_agent",
            "code": "prepare_coding_request"
        }
    )

    builder.add_edge(
        "prepare_coding_request",
        "accept_coding"
    )

    builder.add_conditional_edges(
        "accept_coding",
        lambda state:
            "end"
            if state.get("next_node") == "denied"
            else state["next_node"],
        {
            "coding_agent": "coding_agent",
            "end": END,
            "accept_coding": "prepare_coding_request"
        }
    )

    builder.add_edge("chat_agent", END)
    builder.add_edge("rag_agent", END)
    builder.add_edge("coding_agent", END)

    checkpointer = InMemorySaver()

    return builder.compile(
        checkpointer=checkpointer
    )