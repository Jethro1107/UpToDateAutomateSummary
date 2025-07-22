from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Union, Any

class NoteItem(BaseModel):
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
        if self.subitems and not self.text.endswith(":"):
            self.text += ":"

class NoteSection(BaseModel):
    title: str = Field(description="Title of the note section (e.g., 'Pathophysiology').")
    content: List[NoteItem] = Field(
        description="List of note items with text, optional subitems, sources, quotes, and reasoning."
    )
    source: str = Field(description="Primary source JSON file for the section.")