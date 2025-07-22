ORCHESTRATOR_PROMPT = """You are an expert medical note generator tasked with creating structured notes for medical topics in a Zettelkasten format. Your goal is to generate a list of sections for a medical note on the topic '{topic}' of type '{note_type}' (e.g., 'condition' or 'complaint'). Each section should have a title and a recommended structure (e.g., 'Simple list', 'Detailed hierarchy').

Return a JSON object with a 'sections' key containing an array of section objects, each with 'title' and 'structure' fields. For example:
{
  "sections": [
    {"title": "Definition", "structure": "Simple list"},
    {"title": "Epidemiology", "structure": "Simple list"},
    {"title": "Treatment", "structure": "Detailed hierarchy"}
  ]
}

Based on the topic '{topic}' and note type '{note_type}', suggest appropriate sections and their structures.
"""

HIGH_LEVEL_PROMPT = """Generate a high-level summary for the '{section}' section of a medical note on '{topic}' in {output_format} format. Use the structure '{structure}' (e.g., 'Simple list'). Base the summary on the provided data:
{data}

Return a JSON object conforming to the NoteSection schema with title, content (list of NoteItem or strings), and source.
"""

GAP_EVALUATOR_PROMPT = """Evaluate the summary for the '{section}' section of a medical note on '{topic}' with structure '{structure}'. Identify gaps or missing information based on the summary:
{summary}
and the original data:
{data}

Return a JSON object with a 'gaps' key containing an array of gap objects, each with 'missing', 'query', and 'reasoning' fields.
"""

DETAIL_QUERY_PROMPT = """Generate detailed subitems for the focus point '{focus}' in the '{section}' section of a medical note on '{topic}'. Use the provided data:
{data}

Return a JSON array of NoteItem objects with text, source, quote, reasoning, and subitems fields.
"""

OPTIMIZER_PROMPT = """Optimize the summary for the '{section}' section of a medical note on '{topic}' in {output_format} format with structure '{structure}'. Incorporate identified gaps:
{gaps}
Original summary:
{summary}
Additional details:
{details}
Original data:
{data}

Return a JSON object conforming to the NoteSection schema with title, content (list of NoteItem or strings), and source.
"""