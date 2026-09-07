import os
import subprocess
from config import CLAUDE_PATH, WORKSPACE_DIR, llm
from config import llm
from state import State


def prepare_coding_request(state: State):
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the latest coding request into "
                "a clear instruction for Claude Code. Use the conversation history as context. Only output the instruction, no explanation"
            )
        }
    ] + state["messages"]

    response = llm.invoke(messages)

    return {
        "messages": [
            {
                "role": "user",
                "content": response.content
            }
        ]
    }


def prompt_llm_code(state: State):
    user_prompt = state["messages"][-1].content

    result = subprocess.run(
        [
            str(CLAUDE_PATH),
            "-p",
            user_prompt,
            "--permission-mode",
            "acceptEdits"
        ],
        cwd=str(WORKSPACE_DIR),
        capture_output=True,
        text=True
    )

    output = (
        result.stdout.strip()
        or result.stderr.strip()
    )

    return {
        "messages": [
            {
                "role": "assistant",
                "content": output
            }
        ]
    }