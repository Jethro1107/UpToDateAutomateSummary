from typing import Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field

class NoteItem(BaseModel):
    """A single note item with text, optional subitems, and metadata."""
    text: str = Field(description="Text content of a bullet point or subheading, with *bold* or _underscore_ for emphasis. Appends a colon (:) if subitems are present.")
    subitems: List[Union['NoteItem', str]] = Field(
        default_factory=list,
        description="Subitems as strings (simple bullet points) or NoteItem objects (nested subheadings). Supports multiple nesting levels."
    )
    source: Optional[str] = Field(
        default=None,
        description="Source reference (e.g., JSON file name) for this item."
    )
    quote: Optional[str] = Field(
        default=None,
        description="Direct quote from the source supporting this item."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="AI reasoning for including this item or querying details (e.g., 'Queried due to missing dosing information')."
    )

    def model_post_init(self, __context: Any) -> None:
        """Append a colon to text if subitems exist."""
        if self.subitems and not self.text.endswith(":"):
            self.text += ":"

class NoteSection(BaseModel):
    """A section of the medical note with a title and content."""
    title: str = Field(description="Title of the note section (e.g., 'Pathophysiology').")
    content: List[NoteItem] = Field(
        description="List of note items with text, optional subitems, sources, quotes, and reasoning."
    )
    source: str = Field(description="Primary source JSON file for the section.")

class WorkflowState(BaseModel):
    """State for the LangGraph workflow."""
    topic: str = Field(description="Medical topic (e.g., 'Hypertension').")
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
