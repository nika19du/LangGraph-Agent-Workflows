def save_graph_image(graph, output_path="graph.png"):
    graph.get_graph().draw_mermaid_png(
        output_file_path=output_path
    )