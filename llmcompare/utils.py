"""Utility functions for llmcompare."""

import json
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, data: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file.
    
    Each dictionary is written as a JSON object on a separate line.
    
    Args:
        path: Path to the output JSONL file
        data: List of dictionaries to write
        
    Example:
        >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        >>> write_jsonl("people.jsonl", data)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries.
    
    Each line is parsed as a JSON object.
    
    Args:
        path: Path to the input JSONL file
        
    Returns:
        List of dictionaries, one per line in the file
        
    Example:
        >>> data = read_jsonl("people.jsonl")
        >>> print(data)
        [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    """
    path = Path(path)
    data = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    return data

