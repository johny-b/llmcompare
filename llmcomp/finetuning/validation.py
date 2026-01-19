"""Validation for OpenAI finetuning files."""

import json
import os
from dataclasses import dataclass, field

# Valid roles for OpenAI finetuning
VALID_ROLES = {"system", "user", "assistant", "tool"}

# Allowed keys per role
ALLOWED_KEYS_BY_ROLE = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {"role", "content", "name", "weight", "tool_calls"},
    "tool": {"role", "content", "tool_call_id"},
}

# Minimum number of examples required by OpenAI
MIN_EXAMPLES = 10


@dataclass
class ValidationError:
    """A single validation error found in a finetuning file."""

    line: int  # 1-based line number (0 for file-level errors)
    message: str

    def __str__(self):
        if self.line == 0:
            return self.message
        return f"Line {self.line}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a finetuning file."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    num_examples: int = 0

    def __str__(self):
        if self.valid:
            return f"✓ Valid ({self.num_examples} examples)"

        lines = [f"✗ Invalid file ({len(self.errors)} error(s))"]
        for error in self.errors[:10]:  # Show first 10 errors
            lines.append(f"  {error}")
        if len(self.errors) > 10:
            lines.append(f"  ... and {len(self.errors) - 10} more errors")
        return "\n".join(lines)


def validate_finetuning_file(file_name: str) -> ValidationResult:
    """Validate a JSONL file for OpenAI finetuning.

    Checks:
    - File is valid JSONL (one JSON object per line)
    - At least 10 examples (OpenAI requirement)
    - Each example has a 'messages' array
    - Messages have valid 'role' (system, user, assistant, tool)
    - Messages only contain allowed keys for their role:
        - system/user: role, content, name
        - assistant: role, content, name, weight, tool_calls
        - tool: role, content, tool_call_id
    - Messages have valid 'content' (string or array for multimodal)
    - Each example has at least one 'assistant' message
    - Last message must be from 'assistant'
    - 'weight' field (assistant only): must be 0 or 1, last assistant cannot be 0
    - 'tool_calls' field (assistant only): validates structure (id, type, function)
    - 'tool' messages require 'tool_call_id'

    Args:
        file_name: Path to the JSONL file to validate.

    Returns:
        ValidationResult with valid=True/False and any errors found.

    Example:
        result = validate_finetuning_file("my_dataset.jsonl")
        if not result.valid:
            for error in result.errors:
                print(error)
    """
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []
    num_examples = 0

    # Check if file exists
    if not os.path.exists(file_name):
        errors.append(ValidationError(0, f"File not found: {file_name}"))
        result = ValidationResult(valid=False, errors=errors, num_examples=0)
        return result

    # Read and validate each line
    with open(file_name, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            num_examples += 1
            line_errors = _validate_line(line, line_num)
            errors.extend(line_errors)

    # Check minimum examples
    if num_examples < MIN_EXAMPLES:
        errors.append(
            ValidationError(
                0,
                f"File has {num_examples} examples, but OpenAI requires at least {MIN_EXAMPLES}.",
            )
        )

    result = ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        num_examples=num_examples,
    )

    return result


def _validate_line(line: str, line_num: int) -> list[ValidationError]:
    """Validate a single line of the JSONL file."""
    errors: list[ValidationError] = []

    # Parse JSON
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        errors.append(ValidationError(line_num, f"Invalid JSON: {e}"))
        return errors

    if not isinstance(data, dict):
        errors.append(ValidationError(line_num, "Each line must be a JSON object"))
        return errors

    # Check for 'messages' key
    if "messages" not in data:
        errors.append(ValidationError(line_num, "Missing 'messages' key"))
        return errors

    messages = data["messages"]
    if not isinstance(messages, list):
        errors.append(ValidationError(line_num, "'messages' must be an array"))
        return errors

    if len(messages) == 0:
        errors.append(ValidationError(line_num, "'messages' array is empty"))
        return errors

    # Validate each message
    has_assistant = False
    last_assistant_idx = -1
    for i, msg in enumerate(messages):
        msg_errors = _validate_message(msg, line_num, i)
        errors.extend(msg_errors)

        if isinstance(msg, dict) and msg.get("role") == "assistant":
            has_assistant = True
            last_assistant_idx = i

    # Check for at least one assistant message
    if not has_assistant:
        errors.append(
            ValidationError(
                line_num,
                "No 'assistant' message found. Each example needs at least one assistant response.",
            )
        )

    # Check last message is from assistant (not user, system, or tool)
    if len(messages) > 0:
        last_msg = messages[-1]
        last_role = last_msg.get("role") if isinstance(last_msg, dict) else None
        if last_role != "assistant":
            errors.append(
                ValidationError(
                    line_num,
                    f"Last message must be from 'assistant', got '{last_role}'.",
                )
            )

    # Check last assistant message doesn't have weight=0
    if last_assistant_idx >= 0:
        last_assistant_msg = messages[last_assistant_idx]
        if isinstance(last_assistant_msg, dict) and last_assistant_msg.get("weight") == 0:
            errors.append(
                ValidationError(
                    line_num,
                    "Last assistant message cannot have weight=0.",
                )
            )

    return errors


def _validate_message(msg: dict, line_num: int, msg_idx: int) -> list[ValidationError]:
    """Validate a single message within an example."""
    errors: list[ValidationError] = []
    prefix = f"messages[{msg_idx}]"

    if not isinstance(msg, dict):
        errors.append(ValidationError(line_num, f"{prefix}: must be an object"))
        return errors

    # Check 'role'
    role = None
    if "role" not in msg:
        errors.append(ValidationError(line_num, f"{prefix}: missing 'role'"))
    else:
        role = msg["role"]
        if role not in VALID_ROLES:
            errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}: invalid role '{role}'. Must be one of: {', '.join(sorted(VALID_ROLES))}",
                )
            )

    # Check for unknown keys (only if role is valid)
    if role in ALLOWED_KEYS_BY_ROLE:
        allowed_keys = ALLOWED_KEYS_BY_ROLE[role]
        unknown_keys = set(msg.keys()) - allowed_keys
        if unknown_keys:
            errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}: unknown key(s) for role '{role}': {', '.join(sorted(unknown_keys))}. "
                    f"Allowed: {', '.join(sorted(allowed_keys))}",
                )
            )

    # Check 'content'
    if "content" not in msg:
        # Content can be omitted if there's a tool_calls field (assistant only)
        if role != "assistant" or "tool_calls" not in msg:
            errors.append(ValidationError(line_num, f"{prefix}: missing 'content'"))
    else:
        content = msg["content"]
        content_errors = _validate_content(content, line_num, prefix, role)
        errors.extend(content_errors)

    # Role-specific validation
    if role == "assistant":
        errors.extend(_validate_assistant_message(msg, line_num, prefix))
    elif role == "tool":
        errors.extend(_validate_tool_message(msg, line_num, prefix))

    return errors


def _validate_assistant_message(msg: dict, line_num: int, prefix: str) -> list[ValidationError]:
    """Validate assistant-specific fields."""
    errors: list[ValidationError] = []

    # Check 'weight' field
    if "weight" in msg:
        weight = msg["weight"]
        if weight not in (0, 1):
            errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}: 'weight' must be 0 or 1, got {weight!r}",
                )
            )

    # Check 'tool_calls' field
    if "tool_calls" in msg:
        tool_calls = msg["tool_calls"]
        if not isinstance(tool_calls, list):
            errors.append(ValidationError(line_num, f"{prefix}: 'tool_calls' must be an array"))
        elif len(tool_calls) == 0:
            errors.append(ValidationError(line_num, f"{prefix}: 'tool_calls' array is empty"))
        else:
            for i, tc in enumerate(tool_calls):
                errors.extend(_validate_tool_call(tc, line_num, f"{prefix}.tool_calls[{i}]"))

    return errors


def _validate_tool_call(tc: dict, line_num: int, prefix: str) -> list[ValidationError]:
    """Validate a single tool_call object."""
    errors: list[ValidationError] = []

    if not isinstance(tc, dict):
        errors.append(ValidationError(line_num, f"{prefix}: must be an object"))
        return errors

    # Required fields: id, type, function
    if "id" not in tc:
        errors.append(ValidationError(line_num, f"{prefix}: missing 'id'"))
    elif not isinstance(tc["id"], str):
        errors.append(ValidationError(line_num, f"{prefix}: 'id' must be a string"))

    if "type" not in tc:
        errors.append(ValidationError(line_num, f"{prefix}: missing 'type'"))
    elif tc["type"] != "function":
        errors.append(ValidationError(line_num, f"{prefix}: 'type' must be 'function'"))

    if "function" not in tc:
        errors.append(ValidationError(line_num, f"{prefix}: missing 'function'"))
    elif not isinstance(tc["function"], dict):
        errors.append(ValidationError(line_num, f"{prefix}: 'function' must be an object"))
    else:
        func = tc["function"]
        if "name" not in func:
            errors.append(ValidationError(line_num, f"{prefix}.function: missing 'name'"))
        elif not isinstance(func["name"], str):
            errors.append(ValidationError(line_num, f"{prefix}.function: 'name' must be a string"))

        if "arguments" not in func:
            errors.append(ValidationError(line_num, f"{prefix}.function: missing 'arguments'"))
        elif not isinstance(func["arguments"], str):
            errors.append(ValidationError(line_num, f"{prefix}.function: 'arguments' must be a string"))

    return errors


def _validate_tool_message(msg: dict, line_num: int, prefix: str) -> list[ValidationError]:
    """Validate tool message fields."""
    errors: list[ValidationError] = []

    if "tool_call_id" not in msg:
        errors.append(ValidationError(line_num, f"{prefix}: missing 'tool_call_id'"))
    elif not isinstance(msg["tool_call_id"], str):
        errors.append(ValidationError(line_num, f"{prefix}: 'tool_call_id' must be a string"))

    return errors


def _validate_content(
    content, line_num: int, prefix: str, role: str | None
) -> list[ValidationError]:
    """Validate message content."""
    errors: list[ValidationError] = []

    # Content can be a string
    if isinstance(content, str):
        return errors

    # Content can be None (for assistant messages with tool_calls)
    if content is None:
        return errors

    # Content can be an array (for vision/multimodal)
    if isinstance(content, list):
        for i, part in enumerate(content):
            part_errors = _validate_content_part(part, line_num, f"{prefix}.content[{i}]", role)
            errors.extend(part_errors)
        return errors

    errors.append(
        ValidationError(
            line_num, f"{prefix}: 'content' must be a string or array, got {type(content).__name__}"
        )
    )
    return errors


def _validate_content_part(
    part, line_num: int, prefix: str, role: str | None
) -> list[ValidationError]:
    """Validate a single content part (for multimodal content)."""
    errors: list[ValidationError] = []

    if not isinstance(part, dict):
        errors.append(ValidationError(line_num, f"{prefix}: must be an object"))
        return errors

    if "type" not in part:
        errors.append(ValidationError(line_num, f"{prefix}: missing 'type'"))
        return errors

    part_type = part["type"]

    if part_type == "text":
        if "text" not in part:
            errors.append(ValidationError(line_num, f"{prefix}: missing 'text' for type='text'"))
        elif not isinstance(part["text"], str):
            errors.append(ValidationError(line_num, f"{prefix}: 'text' must be a string"))

    elif part_type == "image_url":
        if "image_url" not in part:
            errors.append(
                ValidationError(line_num, f"{prefix}: missing 'image_url' for type='image_url'")
            )
        elif not isinstance(part["image_url"], dict):
            errors.append(ValidationError(line_num, f"{prefix}: 'image_url' must be an object"))
        elif "url" not in part["image_url"]:
            errors.append(ValidationError(line_num, f"{prefix}: 'image_url' missing 'url'"))

        # Assistant messages cannot contain images
        if role == "assistant":
            errors.append(
                ValidationError(
                    line_num,
                    f"{prefix}: assistant messages cannot contain images",
                )
            )

    return errors
