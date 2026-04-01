"""Detached worker process for fire-and-forget Tinker finetuning.

Spawned by ``FinetuningManager.create_job(blocking=False)`` via
``python -m llmcomp.finetuning.tinker_worker <run_dir>``.

The worker reads training parameters from ``<run_dir>/params.json``,
runs the same ``run_tinker_finetune`` used by the blocking path, and
keeps ``<run_dir>/status.json`` up-to-date so that ``llmcomp-update-jobs``
can report progress.

The API key is passed via the ``TINKER_API_KEY`` environment variable
(never written to disk).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone

from llmcomp.finetuning.params import TinkerTrainingParams
from llmcomp.finetuning.tinker_finetune import run_tinker_finetune


def _write_status(path: str, data: dict):
    """Atomically write a status JSON file (write-to-tmp + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main():
    run_dir = sys.argv[1]

    with open(os.path.join(run_dir, "params.json")) as f:
        params_dict = json.load(f)

    api_key = os.environ["TINKER_API_KEY"]
    data_dir = params_dict.pop("data_dir")
    file_md5 = params_dict.pop("file_md5")
    job_id = params_dict.pop("job_id")
    original_file_name = params_dict.pop("original_file_name")

    params = TinkerTrainingParams(api_key=api_key, **params_dict)

    status_file = os.path.join(run_dir, "status.json")

    # Read started_at once from the status file written by the parent
    # process before spawning us.  Fall back to "now" if missing.
    try:
        with open(status_file) as f:
            started_at = json.load(f).get("started_at")
    except (FileNotFoundError, json.JSONDecodeError):
        started_at = None
    if started_at is None:
        started_at = datetime.now(timezone.utc).isoformat()

    def on_step(step: int, total_steps: int, loss: float):
        _write_status(status_file, {
            "job_id": job_id,
            "suffix": params.suffix,
            "base_model": params.base_model,
            "status": "running",
            "pid": os.getpid(),
            "step": step,
            "total_steps": total_steps,
            "last_loss": loss,
            "started_at": started_at,
        })

    try:
        _write_status(status_file, {
            "job_id": job_id,
            "suffix": params.suffix,
            "base_model": params.base_model,
            "status": "running",
            "pid": os.getpid(),
            "step": 0,
            "total_steps": None,
            "last_loss": None,
            "started_at": started_at,
        })

        model_path = run_tinker_finetune(
            params, data_dir=data_dir, file_md5=file_md5, on_step=on_step,
            record_file_name=original_file_name,
        )

        _write_status(status_file, {
            "job_id": job_id,
            "suffix": params.suffix,
            "base_model": params.base_model,
            "status": "succeeded",
            "pid": os.getpid(),
            "model_path": model_path,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        _write_status(status_file, {
            "job_id": job_id,
            "suffix": params.suffix,
            "base_model": params.base_model,
            "status": "failed",
            "pid": os.getpid(),
            "error": f"{type(e).__name__}: {e}",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
