from langgraph.types import interrupt

from state import State


def accept_coding(state: State):
    user_prompt = state["messages"][-1].content

    decision = interrupt(
        f"About to run Claude Code with request:\n\n"
        f"{user_prompt}\n\n"
        f"Approve? (yes/no, or type a revised request)"
    )

    original_text = str(decision).strip()
    text = original_text.lower()

    if text in ["y", "yes", "approve", "ok"]:
        return {
            "next_node": "coding_agent"
        }

    if text in ["n", "no", "deny", "cancel"]:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Coding request was denied by the user."
                }
            ],
            "next_node": "denied"
        }

    return {
        "messages": [
            {
                "role": "user",
                "content": original_text
            }
        ],
        "next_node": "accept_coding"
    }