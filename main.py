import uuid

from langgraph.types import Command

from graph.builder import build_graph

from graph.visualization import save_graph_image


graph = build_graph()

save_graph_image(graph)

config = {
    "configurable": {
        "thread_id": str(uuid.uuid4())
    }
}

while True:
    user_message = input("Enter a message: ")

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        },
        config=config
    )

    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value

        decision = input(
            f"{prompt}\n"
        )

        result = graph.invoke(
            Command(resume=decision),
            config=config
        )

    print(
        result["messages"][-1].content
    )