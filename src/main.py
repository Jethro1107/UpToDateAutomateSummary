import os
import argparse
import chromadb
from langgraph.graph import StateGraph, START, END
from src.models import WorkflowState
from src.utils.json_loader import load_json_files
from src.utils.embeddings import store_embeddings
from src.nodes import orchestrator, retrieve_docs, worker_node, generate_output

def create_workflow():
    """
    Create and compile the LangGraph workflow.

    Returns:
        compiled workflow: Configured LangGraph workflow.
    """
    workflow = StateGraph(WorkflowState)
    # Pass collection via config["configurable"]["collection"] to match LangGraph's structure
    workflow.add_node("orchestrator", lambda state, config: orchestrator(state, config["configurable"]["collection"]))
    workflow.add_node("retrieve_docs", lambda state, config: retrieve_docs(state, config["configurable"]["collection"]))
    workflow.add_node("process_sections", lambda state, config: process_sections(state, config["configurable"]["collection"]))
    workflow.add_node("generate_output", generate_output)

    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "process_sections")
    workflow.add_edge("process_sections", "generate_output")
    workflow.add_edge("generate_output", END)

    return workflow.compile()

def process_sections(state: WorkflowState, collection: chromadb.Collection) -> WorkflowState:
    """
    Process sections by invoking worker_node for each section.

    Args:
        state (WorkflowState): Current workflow state.
        collection (chromadb.Collection): ChromaDB collection.

    Returns:
        WorkflowState: Updated state with processed sections.
    """
    print(f"DEBUG: Processing sections with {len(state.retrieved_docs)} sections")
    try:
        sections = []
        for section in state.retrieved_docs:
            print(f"DEBUG: Processing section '{section}'")
            section_result = worker_node(state, section, collection)
            sections.append(section_result)
        return WorkflowState(
            topic=state.topic,
            note_type=state.note_type,
            output_format=state.output_format,
            retrieved_docs=state.retrieved_docs,
            sections=sections,
            section_structures=state.section_structures
        )
    except Exception as e:
        print(f"Error in process_sections: {str(e)}")
        raise

def main(topic: str, note_type: str, json_path: str, output_format: str = "markdown") -> str | None:
    """
    Run the medical note generation pipeline.

    Args:
        topic (str): Medical topic (e.g., 'Hypertension').
        note_type (str): Type of note ('condition' or 'complaint').
        json_path (str): Path to JSON file or directory.
        output_format (str): Output format ('markdown' or 'org').

    Returns:
        str | None: Generated output or None if an error occurs.
    """
    # Normalize path for Windows
    json_path = os.path.normpath(json_path)
    print(f"DEBUG: Using json_path: {json_path}")

    # Verify json_path exists
    if not os.path.exists(json_path):
        print(f"Error: Path '{json_path}' does not exist")
        return None

    texts, sources = load_json_files(json_path)
    if not texts:
        print("Error: No valid JSON data loaded. Exiting.")
        return None
    
    print(f"DEBUG: Loaded {len(texts)} texts from {len(sources)} sources")
    collection = store_embeddings(texts, sources)
    print(f"DEBUG: Collection initialized with {collection.count()} documents")
    if collection.count() == 0:
        print("Error: No documents in collection. Exiting.")
        return None

    state = WorkflowState(
        topic=topic,
        note_type=note_type,
        output_format=output_format,
        retrieved_docs={},
        sections=[],
        section_structures={}
    )
    try:
        app = create_workflow()
        # Pass collection in config["configurable"] to match LangGraph's structure
        config = {"configurable": {"collection": collection}}
        print(f"DEBUG: Invoking workflow with config: {config}")
        result = app.invoke(state, config=config)
        output = generate_output(result)
        file_ext = "org" if output_format == "org" else "md"
        output_file = f"{topic.replace(' ', '_')}_notes.{file_ext}"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Output written to {output_file}")
        return output
    except Exception as e:
        print(f"Error in workflow execution: {str(e)}")
        raise

if __name__ == "__main__":
    main(
        topic="Hypertension",
        note_type="condition",
        json_path="json_data/",
        output_format="org"
    )