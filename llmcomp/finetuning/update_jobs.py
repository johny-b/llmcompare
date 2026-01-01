#!/usr/bin/env python3
"""Update finetuning jobs.

Usage:
    llmcomp-update-jobs [DATA_DIR]
"""

import argparse
import os
import sys

from llmcomp.finetuning.manager import DEFAULT_DATA_DIR, FinetuningManager


def main():
    parser = argparse.ArgumentParser(description="Update finetuning jobs from OpenAI API.")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help=f"Directory containing jobs.jsonl (default: {DEFAULT_DATA_DIR} if it exists)",
    )
    args = parser.parse_args()

    if args.data_dir is not None:
        data_dir = args.data_dir
    elif os.path.isdir(DEFAULT_DATA_DIR):
        data_dir = DEFAULT_DATA_DIR
    else:
        print(f"Error: Directory '{DEFAULT_DATA_DIR}' not found.", file=sys.stderr)
        print(f"Specify a data directory: llmcomp-update-jobs <DATA_DIR>", file=sys.stderr)
        sys.exit(1)

    FinetuningManager().update_jobs(data_dir=data_dir)


if __name__ == "__main__":
    main()
