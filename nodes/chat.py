from config import llm
from state import State

def prompt_llm_chat(state: State):
    messages = [
        {'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice.'},
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'assistant', 'content': response.content}]}