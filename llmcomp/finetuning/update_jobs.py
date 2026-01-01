#!/usr/bin/env python3
"""Update finetuning jobs.

Usage:
    llmcomp-update-jobs
"""

from llmcomp.finetuning.manager import FinetuningManager


def main():
    FinetuningManager().update_jobs()


if __name__ == "__main__":
    main()

