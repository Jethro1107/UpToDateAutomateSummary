import ollama
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union, Any
from note_types import NoteSection

def structured_llm(model: str, schema: dict):
    def generate_structured(prompt: str) -> dict:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,
            options={"temperature": 0.5}
        )
        return response["message"]["content"]
    return generate_structured

class WorkflowState(BaseModel):
    topic: str = Field(description="Medical topic (e.g., 'Irritable Bowel Syndrome').")
    note_type: Literal["condition", "complaint"] = Field(
        description="Type: 'condition' for diseases or 'complaint' for symptoms."
    )
    output_format: Literal["markdown", "org"] = Field(
        default="markdown",
        description="Output format: 'markdown' or 'org'."
    )
    retrieved_docs: Dict[str, List[Dict[str, str]]] = Field(
        default_factory=dict,
        description="Section titles mapped to lists of retrieved documents (text and source)."
    )
    sections: List[NoteSection] = Field(
        default_factory=list,
        description="Generated note sections with structured content and sources."
    )
    section_structures: Dict[str, str] = Field(
        default_factory=dict,
        description="Section titles mapped to structuring instructions (e.g., 'Nested list by test categories')."
    )