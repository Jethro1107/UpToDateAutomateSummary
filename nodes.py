from pydantic import ValidationError
import chromadb
from ollama import embed
from typing import List
from note_types import NoteSection, WorkflowState, NoteItem
from model_function import structured_llm
from prompt import HIGH_LEVEL_PROMPT, GAP_EVALUATOR_PROMPT, DETAIL_QUERY_PROMPT, OPTIMIZER_PROMPT

def worker_node(state: WorkflowState, section: str, collection: chromadb.Collection, model: str = "llama3.2") -> NoteSection:
    docs = state.retrieved_docs[section]
    doc_texts = [doc["text"] for doc in docs]
    sources = [doc["source"] for doc in docs]
    structure = state.section_structures.get(section, "Simple list")
    
    # 1. High-Level Summarizer
    high_level_llm = structured_llm(model, NoteSection.model_json_schema())
    prompt = HIGH_LEVEL_PROMPT.format(
        section=section,
        topic=state.topic,
        output_format=state.output_format,
        structure=structure,
        data="".join(doc_texts)
    )
    high_level_output = high_level_llm(prompt)
    section_output = NoteSection.model_validate_json(high_level_output)
    
    # 2. Gap Evaluator
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
    gap_result = gap_llm(gap_prompt)
    
    # 3. Detail Generator
    detail_llm = structured_llm(model, {"type": "array", "items": NoteItem.model_json_schema()})
    for item in section_output.content:
        focus_points = [subitem.text if isinstance(subitem, NoteItem) else subitem for subitem in item.subitems]
        for focus in focus_points:
            # Check for gaps related to this focus
            relevant_gaps = [gap for gap in gap_result["gaps"] if focus.lower() in gap["query"].lower()]
            queries = [gap["query"] for gap in relevant_gaps] or [f"{focus} in the context of {section} for {state.topic}"]
            
            for query, gap in [(q, g) for q in queries for g in relevant_gaps if g["query"] == q] or [(queries[0], None)]:
                # Dynamic RAG query
                response = embed(model="mxbai-embed-large", input=query)
                detail_results = collection.query(query_embeddings=[response["embeddings"]], n_results=3)
                detail_docs = [
                    {"text": doc, "source": meta["source"]}
                    for doc, meta in zip(detail_results["documents"][0], detail_results["metadatas"][0])
                ]
                detail_data = "".join(doc["text"] for doc in detail_docs)
                
                # Generate details
                detail_prompt = DETAIL_QUERY_PROMPT.format(
                    section=section,
                    topic=state.topic,
                    focus=focus,
                    data=detail_data
                )
                detail_subitems = detail_llm(detail_prompt)
                
                # Update subitems with reasoning
                for subitem in detail_subitems:
                    if gap:
                        subitem["reasoning"] = gap["reasoning"]
                
                # Update section output
                for sub in item.subitems:
                    if isinstance(sub, NoteItem) and sub.text == focus:
                        sub.subitems = [NoteItem.model_validate_json(s) for s in detail_subitems]
    
    # 4. Optimizer
    opt_llm = structured_llm(model, NoteSection.model_json_schema())
    opt_prompt = OPTIMIZER_PROMPT.format(
        section=section,
        topic=state.topic,
        output_format=state.output_format,
        structure=structure,
        gaps=gap_result["gaps"],
        summary=high_level_output,
        details=section_output.model_dump_json(),
        data="".join(doc_texts)
    )
    final_output = opt_llm(opt_prompt)
    
    try:
        return NoteSection.model_validate_json(final_output)
    except ValidationError:
        return section_output