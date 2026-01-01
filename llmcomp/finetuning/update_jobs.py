#!/usr/bin/env python3
"""Update finetuning jobs.

Usage:
    python -m llmcomp.finetuning.update_jobs
    
Or via console script (after pip install):
    llmcomp-update-jobs
"""

from llmcomp.finetuning.manager import FinetuningManager


def main():
    FinetuningManager().update_jobs()


if __name__ == "__main__":
    main()

