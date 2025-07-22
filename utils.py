import chromadb
from ollama import embed
import os
import json
from typing import List
from note_types import NoteSection
from model_function import WorkflowState, structured_llm

from prompt import ORCHESTRATOR_PROMPT
def load_json_files(directory: str) -> tuple[List[str], List[str]]:
    texts = []
    sources = []
    for file in os.listdir(directory):
        if file.endswith(".json"):
            with open(os.path.join(directory, file), "r") as f:
                data = json.load(f)
                texts.extend([item["content"] for item in data if "content" in item])
                sources.extend([file] * len([item for item in data if "content" in item]))
    return texts, sources

def store_embeddings(texts: List[str], sources: List[str]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="notes")
    for i, (text, source) in enumerate(zip(texts, sources)):
        response = embed(model="mxbai-embed-large", input=text)
        collection.add(embeddings=[response["embeddings"]], documents=[text], metadatas={"source": source}, ids=[f"doc_{i}"])
    return collection

def orchestrator(state: WorkflowState, collection: chromadb.Collection) -> WorkflowState:
    structured_llm_gen = structured_llm("llama3.2", {
        "type": "object",
        "properties": {"sections": {"type": "array", "items": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "structure": {"type": "string"}}
        }}}
    })
    prompt = ORCHESTRATOR_PROMPT.format(topic=state.topic, note_type=state.note_type)
    result = structured_llm_gen(prompt)
    
    topic_prompt = f"Information about {state.topic}"
    response = embed(model="mxbai-embed-large", input=topic_prompt)
    results = collection.query(query_embeddings=[response["embeddings"]], n_results=10)
    state.retrieved_docs = {
        section["title"]: [{"text": doc, "source": meta["source"]} for doc, meta in zip(results["documents"][0], results["metadatas"][0])]
        for section in result["sections"]
    }
    state.sections = [NoteSection(title=section["title"], content=[], source="Unknown") for section in result["sections"]]
    state.section_structures = {section["title"]: section["structure"] for section in result["sections"]}
    return state

def retrieve_docs(state: WorkflowState, collection: chromadb.Collection) -> WorkflowState:
    for section in state.retrieved_docs:
        prompt = f"{section} of {state.topic}"
        response = embed(model="mxbai-embed-large", input=prompt)
        section_docs = state.retrieved_docs[section]
        section_embeddings = [embed(model="mxbai-embed-large", input=doc["text"])["embeddings"] for doc in section_docs]
        if section_embeddings:
            results = collection.query(
                query_embeddings=[response["embeddings"]],
                include=["documents", "metadatas"],
                n_results=5,
                where={"source": {"$in": [doc["source"] for doc in section_docs]}}
            )
            state.retrieved_docs[section] = [
                {"text": doc, "source": meta["source"]}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
    return state