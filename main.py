from langgraph.graph import StateGraph, START, END
from model_function import WorkflowState
from utils import load_json_files, store_embeddings, orchestrator, retrieve_docs
from output_nodes import generate_output
from nodes import worker_node 

workflow = StateGraph(WorkflowState)
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("retrieve_docs", retrieve_docs)
workflow.add_node("process_sections", lambda state, config: WorkflowState(
    topic=state.topic,
    note_type=state.note_type,
    output_format=state.output_format,
    retrieved_docs=state.retrieved_docs,
    sections=[worker_node(state, section, config["collection"]) for section in state.retrieved_docs],
    section_structures=state.section_structures
))
workflow.add_node("generate_output", generate_output)

workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", "retrieve_docs")
workflow.add_edge("retrieve_docs", "process_sections")
workflow.add_edge("process_sections", "generate_output")
workflow.add_edge("generate_output", END)

app = workflow.compile()

def main(topic: str, note_type: str, json_dir: str, output_format: str = "markdown"):
    texts, sources = load_json_files(json_dir)
    collection = store_embeddings(texts, sources)
    state = WorkflowState(
        topic=topic,
        note_type=note_type,
        output_format=output_format,
        retrieved_docs={},
        sections=[],
        section_structures={}
    )
    try:
        result = app.invoke(state, {"orchestrator": {"collection": collection}, "retrieve_docs": {"collection": collection}, "process_sections": {"collection": collection}})
        output = generate_output(result)
        file_ext = "org" if output_format == "org" else "md"
        with open(f"{topic.replace(' ', '_')}_notes.{file_ext}", "w") as f:
            f.write(output)
        return output
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    main(topic="Irritable Bowel Syndrome", note_type="condition", json_dir="json_files", output_format="org")