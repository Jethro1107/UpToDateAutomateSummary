import os
import json
from typing import List, Tuple

def load_json_files(json_path: str) -> Tuple[List[str], List[str]]:
    """
    Load text content and sources from JSON files in a directory or single file.

    Args:
        json_path (str): Path to a JSON file or directory containing JSON files.

    Returns:
        Tuple[List[str], List[str]]: Lists of text content and corresponding source filenames.

    Raises:
        ValueError: If json_path is neither a valid file nor directory.
    """
    texts = []
    sources = []
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "cp950"]

    def process_file(file_path: str) -> None:
        if os.path.getsize(file_path) == 0:
            print(f"Warning: Empty JSON file '{file_path}' skipped")
            return
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    data = json.load(f)
                    if not data:
                        print(f"Warning: No data in JSON file '{file_path}'")
                        return
                    # Handle UpToDate JSON structure with "content.markdown"
                    if isinstance(data, dict) and "content" in data and isinstance(data["content"], dict) and "markdown" in data["content"]:
                        markdown = data["content"]["markdown"]
                        if isinstance(markdown, str):
                            texts.append(markdown)
                            sources.append(os.path.basename(file_path))
                            print(f"Successfully processed '{file_path}' with {encoding} encoding (content.markdown structure)")
                            return
                        else:
                            print(f"Error: 'markdown' field in '{file_path}' is not a string, got {type(markdown)}")
                    # Fallback for other structures
                    elif isinstance(data, dict):
                        if "content" in data and isinstance(data["content"], str):
                            texts.append(data["content"])
                            sources.append(os.path.basename(file_path))
                            print(f"Successfully processed '{file_path}' with {encoding} encoding (dict with content)")
                            return
                        elif "text" in data and isinstance(data["text"], str):
                            texts.append(data["text"])
                            sources.append(os.path.basename(file_path))
                            print(f"Successfully processed '{file_path}' with {encoding} encoding (dict with text)")
                            return
                        elif "data" in data and isinstance(data["data"], list):
                            content_items = []
                            for item in data["data"]:
                                if isinstance(item, dict):
                                    if "markdown" in item:
                                        content_items.append(item["markdown"])
                                    elif "content" in item:
                                        content_items.append(item["content"])
                                    elif "text" in item:
                                        content_items.append(item["text"])
                                elif isinstance(item, str):
                                    content_items.append(item)
                            if content_items:
                                texts.extend([item for item in content_items if isinstance(item, str)])
                                sources.extend([os.path.basename(file_path)] * len(content_items))
                                print(f"Successfully processed '{file_path}' with {encoding} encoding (nested data list)")
                                return
                        else:
                            print(f"Error: No valid text field ('markdown', 'content', or 'text') in dictionary structure in '{file_path}'")
                    elif isinstance(data, list):
                        content_items = []
                        for item in data:
                            if isinstance(item, dict):
                                if "markdown" in item:
                                    content_items.append(item["markdown"])
                                elif "content" in item:
                                    content_items.append(item["content"])
                                elif "text" in item:
                                    content_items.append(item["text"])
                            elif isinstance(item, str):
                                content_items.append(item)
                        if content_items:
                            texts.extend([item for item in content_items if isinstance(item, str)])
                            sources.extend([os.path.basename(file_path)] * len(content_items))
                            print(f"Successfully processed '{file_path}' with {encoding} encoding (list structure)")
                            return
                    else:
                        print(f"Error: Unexpected JSON structure in '{file_path}': {type(data)}")
                    return
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON in '{file_path}' with {encoding}: {str(e)}")
                continue
            except UnicodeDecodeError as e:
                print(f"Error: Unicode decode issue in '{file_path}' with {encoding}: {str(e)}")
                continue
            except Exception as e:
                print(f"Error: Unexpected issue in '{file_path}' with {encoding}: {str(e)}")
                continue
        print(f"Error: Could not process '{file_path}' with any encoding")

    if os.path.isfile(json_path) and json_path.endswith(".json"):
        process_file(json_path)
    elif os.path.isdir(json_path):
        for file in os.listdir(json_path):
            if file.endswith(".json"):
                process_file(os.path.join(json_path, file))
    else:
        raise ValueError(f"Error: '{json_path}' is neither a valid JSON file nor a directory")
    
    if not texts:
        print(f"Warning: No valid content extracted from '{json_path}'")
    return texts, sources