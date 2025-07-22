from note_types import NoteItem
from model_function import WorkflowState
from typing import Union

def generate_output(state: WorkflowState) -> str:
    if state.output_format == "markdown":
        return generate_markdown(state)
    return generate_org_mode(state)

def generate_markdown(state: WorkflowState) -> str:
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