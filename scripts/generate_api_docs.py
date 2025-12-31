#!/usr/bin/env python3
"""Generate API documentation in Markdown format.

Uses Python's inspect module to extract docstrings and signatures
from selected classes/functions.

Usage:
    python scripts/generate_api_docs.py

Output:
    docs/api.md
"""

import inspect
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_FILE = PROJECT_ROOT / "docs" / "api.md"


def get_signature(obj) -> str:
    """Get function/method signature as a string."""
    try:
        sig = inspect.signature(obj)
        return str(sig)
    except (ValueError, TypeError):
        return "(...)"


def format_docstring(docstring: str | None) -> str:
    """Format a docstring for markdown output, converting Args/Returns to lists."""
    if not docstring:
        return ""
    
    cleaned = inspect.cleandoc(docstring)
    lines = cleaned.split("\n")
    result = []
    
    in_section = None  # 'args', 'returns', 'raises', 'example'
    current_item = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for section headers
        if stripped in ("Args:", "Arguments:"):
            in_section = "args"
            result.append("\n**Arguments:**\n")
            i += 1
            continue
        elif stripped in ("Returns:", "Return:"):
            in_section = "returns"
            result.append("\n**Returns:**\n")
            i += 1
            continue
        elif stripped in ("Raises:", "Raise:"):
            in_section = "raises"
            result.append("\n**Raises:**\n")
            i += 1
            continue
        elif stripped in ("Example:", "Examples:"):
            in_section = "example"
            result.append("\n**Example:**\n")
            i += 1
            continue
        
        if in_section in ("args", "raises"):
            # Check if this is a new argument (name: description pattern)
            if stripped and not line.startswith(" " * 8) and ":" in stripped:
                # Looks like "arg_name: description" or "arg_name (type): description"
                colon_idx = stripped.find(":")
                arg_part = stripped[:colon_idx].strip()
                desc_part = stripped[colon_idx + 1:].strip()
                
                # Handle **kwargs style
                if arg_part.startswith("**"):
                    result.append(f"- `{arg_part}`: {desc_part}")
                elif "(" in arg_part:
                    # Has type annotation like "name (type)"
                    result.append(f"- `{arg_part}`: {desc_part}")
                else:
                    result.append(f"- `{arg_part}`: {desc_part}")
                current_item = True
            elif stripped.startswith("- ") and current_item:
                # Sub-item in an argument description
                result.append(f"  {stripped}")
            elif stripped and current_item:
                # Continuation of previous item description
                result[-1] += " " + stripped
            elif not stripped:
                # Empty line ends the section
                in_section = None
                current_item = None
                result.append("")
        elif in_section == "returns":
            if stripped:
                if stripped.startswith("- "):
                    result.append(stripped)
                elif stripped.startswith("DataFrame with columns:"):
                    result.append(stripped + "\n")
                else:
                    result.append(stripped)
            else:
                in_section = None
                result.append("")
        elif in_section == "example":
            result.append(line)
        else:
            # Regular text
            if stripped:
                result.append(line)
            else:
                result.append("")
        
        i += 1
    
    return "\n".join(result)


def document_class(cls, method_filter=None) -> str:
    """Generate markdown documentation for a class."""
    lines = []

    # Class header
    module = cls.__module__
    lines.append(f"## `{cls.__name__}`\n")
    lines.append(f"*Full path: `{module}.{cls.__name__}`*\n")

    # Class docstring
    if cls.__doc__:
        lines.append(format_docstring(cls.__doc__))
        lines.append("")

    # Class attributes (for Config-like classes)
    if hasattr(cls, "_defaults") and isinstance(cls._defaults, dict):
        lines.append("### Configuration Options\n")
        lines.append("| Attribute | Default | Description |")
        lines.append("|-----------|---------|-------------|")
        for attr, default in cls._defaults.items():
            lines.append(f"| `{attr}` | `{default!r}` | |")
        lines.append("")

    # Properties (check metaclass for properties like url_key_pairs)
    property_docs = []
    for name in dir(type(cls)):
        if name.startswith("_"):
            continue
        try:
            prop = getattr(type(cls), name)
            if isinstance(prop, property) and prop.fget and prop.fget.__doc__:
                prop_lines = [f"#### `{name}`\n"]
                prop_lines.append(format_docstring(prop.fget.__doc__))
                prop_lines.append("")
                property_docs.append((name, "\n".join(prop_lines)))
        except AttributeError:
            continue

    if property_docs:
        lines.append("### Properties\n")
        for _, doc in sorted(property_docs):
            lines.append(doc)

    # Methods
    documented = set()
    method_docs = []

    for name in dir(cls):
        if method_filter and not method_filter(name):
            continue
        if name in documented:
            continue

        try:
            attr = getattr(cls, name)
        except AttributeError:
            continue

        if not callable(attr):
            continue

        # Get the actual function
        func = attr.__func__ if hasattr(attr, "__func__") else attr

        try:
            sig = get_signature(func)
        except Exception:
            sig = "(...)"

        documented.add(name)
        doc_lines = [f"#### `{name}{sig}`\n"]

        docstring = func.__doc__ if hasattr(func, "__doc__") else None
        if docstring:
            doc_lines.append(format_docstring(docstring))
            doc_lines.append("")

        method_docs.append((name, "\n".join(doc_lines)))

    if method_docs:
        lines.append("### Methods\n")
        # Sort with __init__ first
        method_docs.sort(key=lambda x: (x[0] != "__init__", x[0]))
        for _, doc in method_docs:
            lines.append(doc)

    return "\n".join(lines)


def document_methods(cls, methods: list[str]) -> str:
    """Generate markdown documentation for specific class methods only."""
    lines = []

    # Class header
    module = cls.__module__
    lines.append(f"## `{cls.__name__}`\n")
    lines.append(f"*Full path: `{module}.{cls.__name__}`*\n")

    # Class docstring
    if cls.__doc__:
        lines.append(format_docstring(cls.__doc__))
        lines.append("")

    # Methods
    method_docs = []
    for name in methods:
        try:
            attr = getattr(cls, name)
        except AttributeError:
            continue

        if not callable(attr):
            continue

        func = attr.__func__ if hasattr(attr, "__func__") else attr

        try:
            sig = get_signature(func)
        except Exception:
            sig = "(...)"

        doc_lines = [f"#### `{name}{sig}`\n"]

        docstring = func.__doc__ if hasattr(func, "__doc__") else None
        if docstring:
            doc_lines.append(format_docstring(docstring))
            doc_lines.append("")

        method_docs.append((name, "\n".join(doc_lines)))

    if method_docs:
        lines.append("### Methods\n")
        for _, doc in method_docs:
            lines.append(doc)

    return "\n".join(lines)


def main():
    """Generate API documentation."""
    # Import here to avoid top-level import issues
    from llmcompare.config import Config
    from llmcompare.question.judge import FreeFormJudge, RatingJudge
    from llmcompare.question.question import FreeForm, NextToken, Question, Rating

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# API Reference\n",
        "*Auto-generated from source code docstrings.*\n",
        "---\n",
    ]

    # FreeForm: __init__, df, plot
    print("Documenting FreeForm...")
    lines.append(document_methods(FreeForm, ["__init__", "df", "plot"]))
    lines.append("\n---\n")

    # NextToken: __init__, df, plot
    print("Documenting NextToken...")
    lines.append(document_methods(NextToken, ["__init__", "df", "plot"]))
    lines.append("\n---\n")

    # Rating: __init__, df, plot
    print("Documenting Rating...")
    lines.append(document_methods(Rating, ["__init__", "df", "plot"]))
    lines.append("\n---\n")

    # FreeFormJudge: __init__, get_cache
    print("Documenting FreeFormJudge...")
    lines.append(document_methods(FreeFormJudge, ["__init__", "get_cache"]))
    lines.append("\n---\n")

    # RatingJudge: __init__, get_cache
    print("Documenting RatingJudge...")
    lines.append(document_methods(RatingJudge, ["__init__", "get_cache"]))
    lines.append("\n---\n")

    # Config - all public methods
    print("Documenting Config...")
    lines.append(document_class(Config, lambda name: not name.startswith("_")))
    lines.append("\n---\n")

    # Question.create, Question.load_dict, Question.from_yaml
    print("Documenting Question factory methods...")
    lines.append(document_methods(Question, ["create", "load_dict", "from_yaml"]))
    lines.append("\n---\n")

    OUTPUT_FILE.write_text("\n".join(lines))
    print(f"\n✓ API documentation written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

