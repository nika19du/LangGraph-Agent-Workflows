from typing import Literal

from pydantic import BaseModel, Field

from config import llm
from state import State


class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(
        ...,
        description="Classify whether the user wants to just chat, ask for knowledge or change code in the project."
    )


def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([
        {
            "role": "system",
            'content': 'Determine / classify whether the user wants to chat ("chat"), retrieve knowledge ("knowledge") or change code ("code).',
        },
        {
            "role": "user",
            "content": state["messages"][-1].content
        }
    ])

    return {
        "message_intent": result.message_intent
    }