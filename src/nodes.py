import chromadb
import ollama
import json
from typing import Dict, List, Union
from src.models import WorkflowState, NoteSection, NoteItem
from src.prompts import (
    ORCHESTRATOR_PROMPT,
    HIGH_LEVEL_PROMPT,
    GAP_EVALUATOR_PROMPT,
    DETAIL_QUERY_PROMPT,
    OPTIMIZER_PROMPT
)

def structured_llm(model: str, schema: dict):
    """
    Create a structured LLM function for JSON output.

    Args:
        model (str): LLM model name (e.g., 'llama3.2').
        schema (dict): JSON schema for output validation.

    Returns:
        callable: Function that generates structured output.
    """
    def generate_structured(prompt: str) -> dict:
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format=schema,
                options={"temperature": 0.5}
            )
            content = response["message"]["content"]
            if isinstance(content, str):
                content = json.loads(content)
            return content
        except Exception as e:
            print(f"Error in structured_llm for model {model}: {str(e)}")
            raise
    return generate_structured

def orchestrator(state: WorkflowState, collection: chromadb.Collection) -> WorkflowState:
    """
    Generate a list of sections for the medical note without document retrieval.

    Args:
        state (WorkflowState): Current workflow state.
        collection (chromadb.Collection): ChromaDB collection (unused here but required for workflow).

    Returns:
        WorkflowState: Updated state with sections and section structures.
    """
    print(f"DEBUG: Entering orchestrator for topic '{state.topic}', note_type '{state.note_type}'")
    structured_llm_gen = structured_llm("llama3.2", {
        "type": "object",
        "properties": {"sections": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "structure": {"type": "string"}}
        }}}
    })
    prompt = ORCHESTRATOR_PROMPT.format(topic=state.topic, note_type=state.note_type)
    print(f"DEBUG: Orchestrator prompt: {prompt[:100]}...")
    try:
        result = structured_llm_gen(prompt)
        if not isinstance(result, dict) or "sections" not in result:
            print(f"Error: Invalid orchestrator output: {result}")
            result = {"sections": [
                {"title": "Definition", "structure": "Simple list"},
                {"title": "Epidemiology", "structure": "Simple list"},
                {"title": "Treatment", "structure": "Detailed hierarchy"}
            ]}
    except Exception as e:
        print(f"Error in orchestrator LLM call: {str(e)}")
        result = {"sections": [
            {"title": "Definition", "structure": "Simple list"},
            {"title": "Epidemiology", "structure": "Simple list"},
            {"title": "Treatment", "structure": "Detailed hierarchy"}
        ]}
    
    state.sections = [NoteSection(title=section["title"], content=[], source="Unknown") for section in result["sections"]]
    state.section_structures = {section["title"]: section["structure"] for section in result["sections"]}
    print(f"DEBUG: Orchestrator completed with {len(state.sections)} sections")
    return state

def retrieve_docs(state: WorkflowState, collection: chromadb.Collection) -> WorkflowState:
    """
    Retrieve relevant documents for each section using RAG.

    Args:
        state (WorkflowState): Current workflow state.
        collection (chromadb.Collection): ChromaDB collection for document retrieval.

    Returns:
        WorkflowState: Updated state with retrieved documents.
    """
    print(f"DEBUG: Entering retrieve_docs with collection: {collection}, count: {collection.count()}")
    if not isinstance(collection, chromadb.Collection):
        print(f"Error: Invalid collection type: {type(collection)}")
        raise TypeError("Collection must be a chromadb.Collection")
    
    state.retrieved_docs = {}
    for section in [s.title for s in state.sections]:
        prompt = f"{section} of {state.topic}"
        try:
            response = ollama.embed(model="mxbai-embed-large", input=prompt)
            results = collection.query(
                query_embeddings=[response["embeddings"]],
                include=["documents", "metadatas"],
                n_results=5
            )
            state.retrieved_docs[section] = [
                {"text": doc, "source": meta["source"]}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ] if results["documents"] else []
            print(f"DEBUG: Retrieved {len(state.retrieved_docs[section])} documents for section '{section}'")
        except Exception as e:
            print(f"Error in retrieve_docs for section '{section}': {str(e)}")
            state.retrieved_docs[section] = []
    print(f"DEBUG: retrieve_docs completed with {len(state.retrieved_docs)} sections")
    return state

def worker_node(state: WorkflowState, section: str, collection: chromadb.Collection, model: str = "llama3.2") -> NoteSection:
    """
    Process a single section using RAG for summarization, gap evaluation, and optimization.

    Args:
        state (WorkflowState): Current workflow state.
        section (str): Section title to process.
        collection (chromadb.Collection): ChromaDB collection for document retrieval.
        model (str): LLM model name (default: 'llama3.2').

    Returns:
        NoteSection: Processed section with structured content.
    """
    print(f"DEBUG: Entering worker_node for section '{section}' with collection: {collection}, count: {collection.count()}")
    docs = state.retrieved_docs.get(section, [])
    if not docs:
        print(f"Warning: No documents retrieved for section '{section}'")
        return NoteSection(title=section, content=[], source="Unknown")
    doc_texts = [doc["text"] for doc in docs if isinstance(doc["text"], str)]
    sources = [doc["source"] for doc in docs if isinstance(doc["text"], str)]
    structure = state.section_structures.get(section, "Simple list")
    
    # 1. High-Level Summarizer (RAG)
    high_level_llm = structured_llm(model, NoteSection.model_json_schema())
    prompt = HIGH_LEVEL_PROMPT.format(
        section=section,
        topic=state.topic,
        output_format=state.output_format,
        structure=structure,
        data="".join(doc_texts)
    )
    try:
        high_level_output = high_level_llm(prompt)
        section_output = NoteSection.model_validate_json(high_level_output)
    except Exception as e:
        print(f"Error in high-level summarizer for section '{section}': {str(e)}")
        return NoteSection(title=section, content=[], source="Unknown")
    
    # 2. Gap Evaluator (RAG)
    gap_schema = {"type": "object", "properties": {
        "gaps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "missing": {"type": "string"},
                "query": {"type": "string"},
                "reasoning": {"type": "string"}
            }
        }}
    }}
    gap_llm = structured_llm(model, gap_schema)
    gap_prompt = GAP_EVALUATOR_PROMPT.format(
        section=section,
        topic=state.topic,
        structure=structure,
        summary=section_output.model_dump_json(),
        data="".join(doc_texts)
    )
    try:
        gap_result = gap_llm(gap_prompt)
    except Exception as e:
        print(f"Error in gap evaluator for section '{section}': {str(e)}")
        gap_result = {"gaps": []}
    
    # 3. Detail Generator (RAG)
    detail_llm = structured_llm(model, {"type": "array", "items": NoteItem.model_json_schema()})
    for item in section_output.content:
        focus_points = [subitem.text if isinstance(subitem, NoteItem) else subitem for subitem in item.subitems]
        for focus in focus_points:
            relevant_gaps = [gap for gap in gap_result["gaps"] if focus.lower() in gap["query"].lower()]
            queries = [gap["query"] for gap in relevant_gaps] or [f"{focus} in the context of {section} for {state.topic}"]
            
            for query, gap in [(q, g) for q in queries for g in relevant_gaps if g["query"] == q] or [(queries[0], None)]:
                try:
                    response = ollama.embed(model="mxbai-embed-large", input=query)
                    detail_results = collection.query(query_embeddings=[response["embeddings"]], n_results=3)
                    detail_docs = [
                        {"text": doc, "source": meta["source"]}
                        for doc, meta in zip(detail_results["documents"][0], detail_results["metadatas"][0])
                    ]
                    detail_data = "".join(doc["text"] for doc in detail_docs)
                    
                    detail_prompt = DETAIL_QUERY_PROMPT.format(
                        section=section,
                        topic=state.topic,
                        focus=focus,
                        data=detail_data
                    )
                    detail_subitems = detail_llm(detail_prompt)
                    for subitem in detail_subitems:
                        if gap:
                            subitem["reasoning"] = gap["reasoning"]
                    for sub in item.subitems:
                        if isinstance(sub, NoteItem) and sub.text == focus:
                            sub.subitems = [NoteItem.model_validate_json(json.dumps(s)) for s in detail_subitems]
                except Exception as e:
                    print(f"Error in detail generator for section '{section}', focus '{focus}': {str(e)}")
    
    # 4. Optimizer (RAG)
    opt_llm = structured_llm(model, NoteSection.model_json_schema())
    opt_prompt = OPTIMIZER_PROMPT.format(
        section=section,
        topic=state.topic,
        output_format=state.output_format,
        structure=structure,
        gaps=json.dumps(gap_result["gaps"]),
        summary=high_level_output,
        details=section_output.model_dump_json(),
        data="".join(doc_texts)
    )
    try:
        final_output = opt_llm(opt_prompt)
        return NoteSection.model_validate_json(final_output)
    except Exception as e:
        print(f"Warning: Validation error in optimizer for section '{section}': {str(e)}")
        return section_output

def generate_markdown(state: WorkflowState) -> str:
    """
    Generate markdown output from workflow state.

    Args:
        state (WorkflowState): Workflow state with sections.

    Returns:
        str: Formatted markdown string.
    """
    markdown = f"# Notes on {state.topic}\n\n"
    def format_item(item: Union[NoteItem, str], level: int = 0) -> str:
        indent = "  " * level
        if isinstance(item, str):
            return f"{indent}- {item}\n"
        result = f"{indent}- {item.text}\n"
        if item.source:
            result += f"{indent}  *Source*: {item.source}\n"
        if item.quote:
            result += f"{indent}  *Quote*: {item.quote}\n"
        if item.reasoning:
            result += f"{indent}  *Reasoning*: {item.reasoning}\n"
        for subitem in item.subitems:
            result += format_item(subitem, level + 1)
        return result
    
    for section in state.sections:
        markdown += f"## {section.title}\n<details>\n<summary>View {section.title}</summary>\n\n"
        for item in section.content:
            markdown += format_item(item)
        markdown += f"*Primary Source*: {section.source}\n\n</details>\n\n"
    return markdown

def generate_org_mode(state: WorkflowState) -> str:
    """
    Generate org-mode output from workflow state.

    Args:
        state (WorkflowState): Workflow state with sections.

    Returns:
        str: Formatted org-mode string.
    """
    org = f"* Notes on {state.topic}\n\n"
    def format_item_org(item: Union[NoteItem, str], level: int = 0) -> str:
        indent = "  " * level
        if isinstance(item, str):
            return f"{indent}- {item}\n"
        result = f"{indent}- {item.text}\n"
        if item.source or item.quote or item.reasoning:
            result += f"{indent}  :PROPERTIES:\n"
            if item.source:
                result += f"{indent}  :Source: {item.source}\n"
            if item.quote:
                result += f"{indent}  :Quote: {item.quote}\n"
            if item.reasoning:
                result += f"{indent}  :Reasoning: {item.reasoning}\n"
            result += f"{indent}  :END:\n"
        for subitem in item.subitems:
            result += format_item_org(subitem, level + 1)
        return result
    
    for section in state.sections:
        org += f"** {section.title}\n"
        for item in section.content:
            org += format_item_org(item)
        org += f"  - *Primary Source*: {section.source}\n\n"
    return org

def generate_output(state: WorkflowState) -> str:
    """
    Generate output in the specified format.

    Args:
        state (WorkflowState): Workflow state with output format.

    Returns:
        str: Formatted output string.
    """
    print(f"DEBUG: Generating output in {state.output_format} format")
    if state.output_format == "markdown":
        return generate_markdown(state)
    return generate_org_mode(state)