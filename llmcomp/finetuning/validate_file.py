#!/usr/bin/env python3
"""Validate a finetuning file and estimate training costs.

Usage:
    llmcomp-validate-file <FILE>
"""

import argparse
import json
import sys

import tiktoken

from llmcomp.finetuning.validation import validate_finetuning_file

# Per-message overhead and reply priming tokens (OpenAI chat format)
TOKENS_PER_MESSAGE = 4
TOKENS_PER_REPLY = 2

COST_PER_1M_TOKENS = {
    "gpt-4.1": 25.0,
    "gpt-4.1-mini": 5.0,
    "gpt-4.1-nano": 1.5,
}


def count_tokens(file_name: str) -> int:
    """Count training tokens in a finetuning JSONL file.

    Uses the o200k_base encoding (GPT-4o/4.1 family) and accounts for
    per-message overhead tokens.
    """
    encoding = tiktoken.get_encoding("o200k_base")
    total = 0

    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = example.get("messages", [])
            for message in messages:
                total += TOKENS_PER_MESSAGE
                for key, value in message.items():
                    if isinstance(value, str):
                        total += len(encoding.encode(value))
                    if key == "name":
                        total -= 1
            total += TOKENS_PER_REPLY

    return total


def main():
    parser = argparse.ArgumentParser(description="Validate a finetuning file and estimate training costs.")
    parser.add_argument("file_name", help="Path to the JSONL file to validate")
    args = parser.parse_args()

    # 1. Validation
    result = validate_finetuning_file(args.file_name)
    print(result)

    if not result.valid:
        sys.exit(1)

    # 2. Cost estimation
    tokens = count_tokens(args.file_name)
    print(f"\nTraining tokens: {tokens:,}")
    print(f"\nEstimated cost per epoch (assuming $/1M training tokens):")
    for model, cost in COST_PER_1M_TOKENS.items():
        estimated = tokens * cost / 1_000_000
        print(f"  {model:20s} ${cost:>5.0f}/1M tokens → ${estimated:,.2f}")


if __name__ == "__main__":
    main()
