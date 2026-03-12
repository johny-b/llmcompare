"""Track finetuned Tinker models.

Tinker model metadata is stored in tinker_models.jsonl (one entry per model).
This is the Tinker equivalent of the OpenAI jobs.jsonl → models pipeline.
"""

import os

import pandas as pd
from filelock import FileLock

from llmcomp.utils import read_jsonl, write_jsonl


def get_tinker_models_df(data_dir: str) -> pd.DataFrame:
    """Read tinker_models.jsonl and return a DataFrame.

    The columns are aligned with the OpenAI models DataFrame where possible:
    model, base_model, file_name, file_md5, suffix, batch_size, epochs.

    Tinker-specific columns: learning_rate, lora_rank.
    """
    fname = os.path.join(data_dir, "tinker_models.jsonl")
    try:
        models = read_jsonl(fname)
    except FileNotFoundError:
        return pd.DataFrame()

    if not models:
        return pd.DataFrame()

    return pd.DataFrame(models)


def save_tinker_model(data_dir: str, model_data: dict):
    """Append a model entry to tinker_models.jsonl (file-lock protected)."""
    fname = os.path.join(data_dir, "tinker_models.jsonl")
    lock = FileLock(fname + ".lock")

    with lock:
        try:
            models = read_jsonl(fname)
        except FileNotFoundError:
            models = []

        models.append(model_data)
        write_jsonl(fname, models)
