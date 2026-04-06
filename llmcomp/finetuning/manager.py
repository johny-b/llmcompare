import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone

import openai
import pandas as pd

from llmcomp.finetuning.params import OpenaiTrainingParams, TinkerTrainingParams
from llmcomp.finetuning.tinker_models import get_tinker_models_df
from llmcomp.finetuning.validation import ValidationResult, validate_finetuning_file
from llmcomp.utils import read_jsonl, write_jsonl

DEFAULT_DATA_DIR = "llmcomp_models"


class FinetuningManager:
    """Manage finetuning runs (OpenAI and Tinker).

    * Create FT jobs via `create_job` (pass OpenaiTrainingParams or TinkerTrainingParams)
    * Fetch updates to FT jobs via `update_jobs`
    * Get a list of models via `get_models` or `get_model_list`

    Args:
        data_dir: Directory for storing finetuning data.  Contains
                  jobs.jsonl (OpenAI), tinker_models.jsonl, tinker_runs/,
                  files.jsonl, and models.csv.  Defaults to "llmcomp_models".
    """

    # Cache: api_key -> organization_id
    _org_cache: dict[str, str] = {}

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir

    #########################################################
    # PUBLIC INTERFACE
    def get_model_list(self, **kwargs) -> list[str]:
        return self.get_models(**kwargs)["model"].tolist()

    def get_models(self, **kwargs) -> pd.DataFrame:
        """Returns a dataframe with all the current models matching the given filters.

        Or just all models if there are no filters.

        Example usage:

            models = FinetuningManager().get_models(
                base_model="gpt-4.1-mini-2025-04-14",
                suffix="my-suffix",
            )

        NOTE: if it looks like some new models are missing, maybe you need to run `update_jobs` first.
        """
        all_models = self._get_all_models()

        mask = pd.Series(True, index=all_models.index)
        for col, val in kwargs.items():
            mask &= all_models[col] == val

        filtered_df = all_models[mask].copy()
        return filtered_df

    def update_jobs(self):
        """Fetch the latest information about all the jobs.

        It's fine to run this many times - the data is not overwritten.
        Sends requests only for jobs that don't have a final status yet.

        Usage:

            FinetuningManager().update_jobs()

        Or from command line: llmcomp-update-jobs
        """
        jobs_file = os.path.join(self.data_dir, "jobs.jsonl")
        try:
            jobs = read_jsonl(jobs_file)
        except FileNotFoundError:
            jobs = []

        # Statuses that mean the job is done (no need to check again)
        final_statuses = {"succeeded", "failed", "cancelled"}

        counts = {"running": 0, "succeeded": 0, "failed": 0, "newly_completed": 0}
        jobs_without_key = []

        for job in jobs:
            # Skip jobs that already have a final status
            if job.get("status") in final_statuses:
                if job["status"] == "succeeded":
                    counts["succeeded"] += 1
                else:
                    counts["failed"] += 1  # failed or cancelled
                continue

            # Skip jobs that already have a model (succeeded before we tracked status)
            if job.get("model") is not None:
                counts["succeeded"] += 1
                continue

            # Try all API keys for this organization
            api_keys = self._get_api_keys_for_org(job["organization_id"])
            if not api_keys:
                jobs_without_key.append(job)
                continue

            job_data = None
            api_key = None
            for key in api_keys:
                try:
                    client = openai.OpenAI(api_key=key)
                    job_data = client.fine_tuning.jobs.retrieve(job["id"])
                    api_key = key
                    break
                except Exception:
                    continue

            if job_data is None:
                jobs_without_key.append(job)
                continue

            status = job_data.status
            job["status"] = status

            if status == "succeeded":
                counts["succeeded"] += 1
                counts["newly_completed"] += 1
                print(f"✓ {job['suffix']}: succeeded → {job_data.fine_tuned_model}")

                # Update model
                job["model"] = job_data.fine_tuned_model

                # Update checkpoints
                checkpoints = self._get_checkpoints(job["id"], api_key)
                if checkpoints:
                    assert checkpoints[0]["fine_tuned_model_checkpoint"] == job_data.fine_tuned_model
                    for i, checkpoint in enumerate(checkpoints[1:], start=1):
                        key_name = f"model-{i}"
                        job[key_name] = checkpoint["fine_tuned_model_checkpoint"]

                # Update seed
                if "seed" not in job or job["seed"] == "auto":
                    job["seed"] = job_data.seed

                # Update hyperparameters
                hyperparameters = job_data.method.supervised.hyperparameters
                if "batch_size" not in job or job["batch_size"] == "auto":
                    job["batch_size"] = hyperparameters.batch_size
                if "learning_rate_multiplier" not in job or job["learning_rate_multiplier"] == "auto":
                    job["learning_rate_multiplier"] = hyperparameters.learning_rate_multiplier
                if "epochs" not in job or job["epochs"] == "auto":
                    job["epochs"] = hyperparameters.n_epochs

            elif status in ("failed", "cancelled"):
                counts["failed"] += 1
                error_msg = ""
                if job_data.error and job_data.error.message:
                    error_msg = f" - {job_data.error.message}"
                print(f"✗ {job['suffix']}: {status}{error_msg}")

            else:
                # Still running (validating_files, queued, running)
                counts["running"] += 1
                eta_str = self._format_eta(job_data.estimated_finish)
                if eta_str:
                    real_eta_str = self._format_eta(job_data.estimated_finish, extra_minutes=20)
                    print(f"… {job['suffix']} ({job['base_model']}): {status} (training ETA: {eta_str}, real ETA: {real_eta_str})")
                elif status == "running":
                    phase = self._get_running_phase(job["id"], client)
                    print(f"… {job['suffix']} ({job['base_model']}): {phase}")
                else:
                    print(f"… {job['suffix']} ({job['base_model']}): {status}")

        write_jsonl(jobs_file, jobs)

        # Check detached Tinker training runs
        tinker_counts = self._update_tinker_runs()
        for key in tinker_counts:
            counts[key] = counts.get(key, 0) + tinker_counts[key]

        # Print summary
        print()
        if counts["running"] > 0:
            print(f"Running: {counts['running']}, Succeeded: {counts['succeeded']}, Failed: {counts['failed']}")
        else:
            print(f"All jobs finished. Succeeded: {counts['succeeded']}, Failed: {counts['failed']}")

        if jobs_without_key:
            print(f"\n⚠ {len(jobs_without_key)} job(s) could not be checked (no matching API key):")
            for job in jobs_without_key:
                print(f"  - {job['suffix']} (org: {job['organization_id']})")

        # Regenerate models.csv with any newly completed jobs
        self._get_all_models()

    def create_job(self, params: OpenaiTrainingParams | TinkerTrainingParams, *, blocking: bool = False) -> str | None:
        """Create a new finetuning job.

        Pass ``OpenaiTrainingParams`` for OpenAI or ``TinkerTrainingParams``
        for Tinker.

        By default (``blocking=False``) all jobs are fire-and-forget:
        the call returns quickly and progress is tracked via
        ``llmcomp-update-jobs``.  Set ``blocking=True`` to wait for the
        training to finish in-process (Tinker only).

        Returns the model path for blocking Tinker jobs, or ``None``
        for fire-and-forget jobs.  Use ``update_jobs`` / ``get_models``
        to discover completed models.

        Example (OpenAI, fire-and-forget):

            manager.create_job(OpenaiTrainingParams(
                api_key=os.environ["OPENAI_API_KEY"],
                file_name="dataset.jsonl",
                base_model="gpt-4.1-mini-2025-04-14",
                suffix="my-experiment",
            ))

        Example (Tinker, fire-and-forget):

            manager.create_job(TinkerTrainingParams(
                api_key=os.environ["TINKER_API_KEY"],
                file_name="dataset.jsonl",
                base_model="Qwen/Qwen3-30B-A3B",
                suffix="my-experiment",
            ))

        Example (Tinker, blocking):

            model_path = manager.create_job(TinkerTrainingParams(
                api_key=os.environ["TINKER_API_KEY"],
                file_name="dataset.jsonl",
                base_model="Qwen/Qwen3-30B-A3B",
                suffix="my-experiment",
            ), blocking=True)
        """
        file_md5 = self._get_file_md5(params.file_name)

        if params.suffix is None:
            params.suffix = self._get_default_suffix(params.file_name)

        self._check_suffix_collision(params.suffix, params.file_name, file_md5)

        if isinstance(params, OpenaiTrainingParams):
            if blocking:
                raise NotImplementedError(
                    "Blocking mode is not supported for OpenAI finetuning. "
                    "Use the default fire-and-forget mode and poll with update_jobs()."
                )
            return self._create_openai_job(params, file_md5)
        elif isinstance(params, TinkerTrainingParams):
            if blocking:
                return self._create_tinker_job_blocking(params, file_md5)
            return self._create_tinker_job_detached(params, file_md5)
        else:
            raise TypeError(f"Expected OpenaiTrainingParams or TinkerTrainingParams, got {type(params).__name__}")

    def openai_validate_file(self, file_name: str) -> ValidationResult:
        """Validate a JSONL file for OpenAI finetuning.

        See `llmcomp.finetuning.validate_finetuning_file` for details.
        """
        return validate_finetuning_file(file_name)

    #########################################################
    # PRIVATE METHODS
    @staticmethod
    def _handle_rate_limit(error: openai.RateLimitError, client: openai.OpenAI):
        """When job creation hits the daily rate limit, estimate when a slot will free up."""
        error_msg = str(error)
        print(f"\n✗ Rate limited: {error_msg}")

        # Parse the limit from the error message
        limit_match = re.search(r"maximum of (\d+) fine-tuning requests per day", error_msg)
        if not limit_match:
            return
        limit = int(limit_match.group(1))

        # List recent jobs to find creation times
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        recent_jobs = []
        try:
            for job in client.fine_tuning.jobs.list(limit=100):
                created = datetime.fromtimestamp(job.created_at, tz=timezone.utc)
                if created >= cutoff:
                    recent_jobs.append(created)
                else:
                    break
        except Exception:
            return

        recent_jobs.sort()
        count = len(recent_jobs)

        if count == 0:
            print(f"\n  Limit: {limit}/day, but no jobs found in this project in the last 24h.")
            print(f"  The rate limit is likely hit by jobs in other projects.")
            return

        oldest = recent_jobs[0]
        slot_opens = oldest + timedelta(hours=24)
        wait = slot_opens - now
        wait_hours = int(wait.total_seconds()) // 3600
        wait_minutes = (int(wait.total_seconds()) % 3600) // 60

        if count >= limit:
            print(f"\n  Limit: {limit}/day, this project has {count} jobs in the last 24h.")
            print(f"  Expect a free slot in ~{wait_hours}h {wait_minutes}m.")
        else:
            print(f"\n  Limit: {limit}/day, but this project only has {count} jobs in the last 24h.")
            print(f"  Other projects are using the remaining slots.")
            print(f"  This project will free a slot in ~{wait_hours}h {wait_minutes}m,")
            print(f"  but another project may free one sooner.")

    def _check_suffix_collision(self, suffix: str, file_name: str, file_md5: str):
        """Raise error if suffix is already used with a different file.

        Checks OpenAI jobs (jobs.jsonl), completed Tinker models
        (tinker_models.jsonl), and in-progress detached Tinker runs
        (tinker_runs/*/status.json) so that cross-provider and
        in-flight suffix collisions are caught.
        """
        entries: list[dict] = []

        jobs_file = os.path.join(self.data_dir, "jobs.jsonl")
        try:
            entries.extend(read_jsonl(jobs_file))
        except FileNotFoundError:
            pass

        tinker_file = os.path.join(self.data_dir, "tinker_models.jsonl")
        try:
            entries.extend(read_jsonl(tinker_file))
        except FileNotFoundError:
            pass

        # Also check in-progress detached Tinker runs
        runs_dir = os.path.join(self.data_dir, "tinker_runs")
        if os.path.isdir(runs_dir):
            for job_dir in os.listdir(runs_dir):
                status_file = os.path.join(runs_dir, job_dir, "status.json")
                try:
                    with open(status_file) as f:
                        status = json.load(f)
                    if status.get("status") in ("starting", "running"):
                        entries.append(status)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass

        for entry in entries:
            if entry.get("suffix") != suffix:
                continue

            if entry.get("file_name") != file_name:
                raise ValueError(
                    f"Suffix '{suffix}' is already used with a different file:\n"
                    f"  Existing: {entry['file_name']}\n"
                    f"  New:      {file_name}\n\n"
                    f"Using the same suffix for different datasets makes model names\n"
                    f"ambiguous. Choose a different suffix for this file."
                )

            if entry.get("file_md5") != file_md5:
                raise ValueError(
                    f"Suffix '{suffix}' is already used with file '{file_name}',\n"
                    f"but the file content has changed (different MD5).\n\n"
                    f"If you modified the dataset, use a different suffix to\n"
                    f"distinguish the new models."
                )

    def _create_openai_job(self, params: OpenaiTrainingParams, file_md5: str) -> None:
        """Create a finetuning job on OpenAI (fire-and-forget)."""
        validation_result = self.openai_validate_file(params.file_name)
        if not validation_result.valid:
            print("Invalid training file.")
            print(validation_result)
            return

        if params.validation_file_name is not None:
            validation_result = self.openai_validate_file(params.validation_file_name)
            if not validation_result.valid:
                print("Invalid validation file.")
                print(validation_result)
                return

        organization_id = self._get_organization_id(params.api_key)
        file_id = self._upload_file_if_not_uploaded(params.file_name, params.api_key, organization_id)

        validation_file_id = None
        if params.validation_file_name is not None:
            validation_file_id = self._upload_file_if_not_uploaded(params.validation_file_name, params.api_key, organization_id)

        data = {
            "model": params.base_model,
            "training_file": file_id,
            "seed": params.seed,
            "suffix": params.suffix,
            "method": {
                "type": "supervised",
                "supervised": {
                    "hyperparameters": {
                        "batch_size": params.batch_size,
                        "learning_rate_multiplier": params.lr_multiplier,
                        "n_epochs": params.epochs,
                    }
                },
            },
        }
        if validation_file_id is not None:
            data["validation_file"] = validation_file_id

        client = openai.OpenAI(api_key=params.api_key)
        try:
            response = client.fine_tuning.jobs.create(**data)
        except openai.RateLimitError as e:
            self._handle_rate_limit(e, client)
            return

        job_id = response.id
        fname = os.path.join(self.data_dir, "jobs.jsonl")
        try:
            ft_jobs = read_jsonl(fname)
        except FileNotFoundError:
            ft_jobs = []

        ft_jobs.append(
            {
                "id": job_id,
                "file_name": params.file_name,
                "base_model": params.base_model,
                "suffix": params.suffix,
                "file_id": file_id,
                "epochs": params.epochs,
                "batch_size": params.batch_size,
                "learning_rate_multiplier": params.lr_multiplier,
                "file_md5": file_md5,
                "organization_id": organization_id,
            }
        )
        write_jsonl(fname, ft_jobs)

        print(f"\n✓ Finetuning job created")
        print(f"  Job ID:     {job_id}")
        print(f"  Base model: {params.base_model}")
        print(f"  Suffix:     {params.suffix}")
        print(f"  File:       {params.file_name} (id: {file_id})")
        if validation_file_id is not None:
            print(f"  Validation: {params.validation_file_name} (id: {validation_file_id})")
        print(f"  Epochs:     {params.epochs}, Batch: {params.batch_size}, LR: {params.lr_multiplier}")
        print(f"  Status:     {response.status}")
        print(f"\nRun `llmcomp-update-jobs` to check progress.")

    def _create_tinker_job_blocking(self, params: TinkerTrainingParams, file_md5: str) -> str:
        """Run Tinker finetuning in-process (blocks until complete)."""
        from llmcomp.finetuning.tinker_finetune import run_tinker_finetune

        return run_tinker_finetune(params, data_dir=self.data_dir, file_md5=file_md5)

    def _create_tinker_job_detached(self, params: TinkerTrainingParams, file_md5: str) -> None:
        """Spawn a detached Tinker training process (fire-and-forget)."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"tinker-{timestamp}-{uuid.uuid4().hex[:6]}"
        run_dir = os.path.join(os.path.abspath(self.data_dir), "tinker_runs", job_id)
        os.makedirs(run_dir)

        params_dict = dataclasses.asdict(params)
        del params_dict["api_key"]
        params_dict["original_file_name"] = params.file_name
        params_dict["file_name"] = os.path.abspath(params.file_name)
        params_dict["data_dir"] = os.path.abspath(self.data_dir)
        params_dict["file_md5"] = file_md5
        params_dict["job_id"] = job_id

        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params_dict, f, indent=2)

        started_at = datetime.now(timezone.utc).isoformat()
        status_path = os.path.join(run_dir, "status.json")

        # Write initial status BEFORE spawning the child so the worker
        # can always read started_at, and update_jobs never sees a missing file.
        _write_status_file(status_path, {
            "job_id": job_id,
            "suffix": params.suffix,
            "base_model": params.base_model,
            "file_name": params.file_name,
            "file_md5": file_md5,
            "status": "starting",
            "pid": None,
            "started_at": started_at,
        })

        log_file = open(os.path.join(run_dir, "log.txt"), "w")
        env = {**os.environ, "TINKER_API_KEY": params.api_key, "PYTHONUNBUFFERED": "1"}

        proc = subprocess.Popen(
            [sys.executable, "-m", "llmcomp.finetuning.tinker_worker", run_dir],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        log_file.close()

        # Record the child PID now that the process has started.
        _write_status_file(status_path, {
            "job_id": job_id,
            "suffix": params.suffix,
            "base_model": params.base_model,
            "file_name": params.file_name,
            "file_md5": file_md5,
            "status": "starting",
            "pid": proc.pid,
            "started_at": started_at,
        })

        print(f"\n✓ Tinker finetuning job started (detached)")
        print(f"  Job ID:     {job_id}")
        print(f"  Base model: {params.base_model}")
        print(f"  Suffix:     {params.suffix}")
        print(f"  Run dir:    {run_dir}")
        print(f"\nRun `llmcomp-update-jobs` to check progress.")

    def _update_tinker_runs(self) -> dict[str, int]:
        """Check status of detached Tinker training runs.

        Returns a dict with keys "running", "succeeded", "failed" counting
        the Tinker runs found.
        """
        runs_dir = os.path.join(self.data_dir, "tinker_runs")
        if not os.path.isdir(runs_dir):
            return {"running": 0, "succeeded": 0, "failed": 0}

        counts: dict[str, int] = {"running": 0, "succeeded": 0, "failed": 0}

        for job_id in sorted(os.listdir(runs_dir)):
            run_dir = os.path.join(runs_dir, job_id)
            status_file = os.path.join(run_dir, "status.json")
            if not os.path.isfile(status_file):
                continue

            with open(status_file) as f:
                status = json.load(f)

            state = status["status"]
            suffix = status.get("suffix", job_id)
            base_model = status.get("base_model", "?")

            if state == "succeeded":
                counts["succeeded"] += 1
                if not status.get("reported"):
                    model_path = status.get("model_path", "?")
                    print(f"✓ {suffix}: succeeded → {model_path}")
                    status["reported"] = True
                    _write_status_file(status_file, status)

            elif state == "failed":
                counts["failed"] += 1
                if not status.get("reported"):
                    error = status.get("error", "unknown error")
                    print(f"✗ {suffix}: failed - {error}")
                    status["reported"] = True
                    _write_status_file(status_file, status)

            elif state in ("running", "starting"):
                pid = status.get("pid")
                if pid is not None and not _is_process_alive(pid):
                    log_tail = _read_log_tail(os.path.join(run_dir, "log.txt"))
                    error = f"Process {pid} died unexpectedly"
                    if log_tail:
                        error += f"\n  Last output: {log_tail}"
                    status["status"] = "failed"
                    status["error"] = error
                    status["finished_at"] = datetime.now(timezone.utc).isoformat()
                    _write_status_file(status_file, status)
                    print(f"✗ {suffix}: failed - {error}")
                    counts["failed"] += 1
                else:
                    step = status.get("step", 0)
                    total = status.get("total_steps", "?")
                    loss = status.get("last_loss")
                    loss_str = f", loss: {loss:.4f}" if loss is not None else ""
                    pid_str = f", pid {pid}" if pid is not None else ""
                    print(f"… {suffix} ({base_model}): step {step}/{total}{loss_str}{pid_str}")
                    counts["running"] += 1

        return counts

    def _get_all_models(self) -> pd.DataFrame:
        jobs_fname = os.path.join(self.data_dir, "jobs.jsonl")
        try:
            jobs = read_jsonl(jobs_fname)
        except FileNotFoundError:
            jobs = []

        models = []
        for job in jobs:
            if job.get("model") is None:
                continue

            model_data = {
                "model": job["model"],
                "base_model": job["base_model"],
                "file_name": job["file_name"],
                "file_id": job["file_id"],
                "file_md5": job["file_md5"],
                "suffix": job["suffix"],
                "batch_size": job["batch_size"],
                "learning_rate_multiplier": job["learning_rate_multiplier"],
                "epochs": job["epochs"],
                "seed": job["seed"],
            }
            models.append(model_data)
            for i in range(1, 3):
                key = f"model-{i}"
                if key in job:
                    checkpoint_data = model_data.copy()
                    checkpoint_data["model"] = job[key]
                    checkpoint_data["epochs"] -= i
                    models.append(checkpoint_data)

        df = pd.DataFrame(models)

        # Include Tinker models
        tinker_df = get_tinker_models_df(self.data_dir)
        if not tinker_df.empty:
            df = pd.concat([df, tinker_df], ignore_index=True)

        df.to_csv(os.path.join(self.data_dir, "models.csv"), index=False)
        return df

    def _upload_file_if_not_uploaded(self, file_name, api_key, organization_id):
        files_fname = os.path.join(self.data_dir, "files.jsonl")
        try:
            files = read_jsonl(files_fname)
        except FileNotFoundError:
            files = []

        md5 = self._get_file_md5(file_name)
        client = openai.OpenAI(api_key=api_key)

        for file in files:
            if file["name"] == file_name and file["md5"] == md5 and file["organization_id"] == organization_id:
                # Verify the file actually exists (it might be in a different project)
                # See: https://github.com/johny-b/llmcomp/issues/31
                try:
                    client.files.retrieve(file["id"])
                    print(f"File {file_name} already uploaded. ID: {file['id']}")
                    return file["id"]
                except openai.NotFoundError:
                    # File is in this organization, but in another project
                    pass

        return self._upload_file(file_name, api_key, organization_id)

    def _upload_file(self, file_name, api_key, organization_id):
        try:
            file_id = self._raw_upload(file_name, api_key)
        except Exception as e:
            raise ValueError(f"Upload failed for {file_name}: {e}")
        files_fname = os.path.join(self.data_dir, "files.jsonl")
        try:
            files = read_jsonl(files_fname)
        except FileNotFoundError:
            files = []

        files.append(
            {
                "name": file_name,
                "md5": self._get_file_md5(file_name),
                "id": file_id,
                "organization_id": organization_id,
            }
        )
        write_jsonl(files_fname, files)
        return file_id

    @staticmethod
    def _raw_upload(file_name, api_key):
        client = openai.OpenAI(api_key=api_key)
        with open(file_name, "rb") as f:
            response = client.files.create(file=f, purpose="fine-tune")
        print(f"Uploaded {file_name} → {response.id}")
        return response.id

    @staticmethod
    def _get_default_suffix(file_name: str) -> str:
        base = os.path.basename(file_name).rsplit(".", 1)[0]
        suffix = base.replace("_", "-")
        if len(suffix) > 64:
            suffix = suffix[:64]
        return suffix

    @staticmethod
    def _get_file_md5(file_name):
        try:
            with open(file_name, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except FileNotFoundError:
            raise FileNotFoundError(f"Training file not found: {file_name}")

    @classmethod
    def _get_organization_id(cls, api_key: str) -> str:
        """Get the organization ID for an API key by making a simple API call."""
        if api_key in cls._org_cache:
            return cls._org_cache[api_key]

        client = openai.OpenAI(api_key=api_key)
        
        # Try to list fine-tuning jobs (limit 1) to get org_id from response
        jobs = client.fine_tuning.jobs.list(limit=1)
        if jobs.data:
            org_id = jobs.data[0].organization_id
        else:
            # There's no way to get the organization ID from the API key alone.
            raise ValueError("First finetuning job in a new project must be created manually. See https://github.com/johny-b/llmcomp/issues/42.")

        cls._org_cache[api_key] = org_id
        return org_id

    @classmethod
    def _get_api_keys_for_org(cls, organization_id: str) -> list[str]:
        """Find all API keys that belong to the given organization."""
        matching_keys = []
        for api_key in cls._get_all_api_keys():
            try:
                org_id = cls._get_organization_id(api_key)
                if org_id == organization_id:
                    matching_keys.append(api_key)
            except Exception:
                continue
        return matching_keys

    @staticmethod
    def _get_all_api_keys() -> list[str]:
        """Get all OpenAI API keys from environment (OPENAI_API_KEY and OPENAI_API_KEY_*)."""
        keys = []
        for env_var in os.environ:
            if env_var == "OPENAI_API_KEY" or env_var.startswith("OPENAI_API_KEY_"):
                key = os.environ.get(env_var)
                if key:
                    keys.append(key)
        return keys

    @staticmethod
    def _get_running_phase(job_id: str, client: openai.OpenAI) -> str:
        """Check events to determine if a running job is training or in safety checks."""
        try:
            events = client.fine_tuning.jobs.list_events(job_id, limit=20)
            for event in events:
                if "usage policies" in event.message.lower():
                    return "safety checks (training done)"
            return "running (no ETA yet)"
        except Exception:
            return "running (no ETA yet)"

    @staticmethod
    def _format_eta(estimated_finish, extra_minutes: int = 0) -> str | None:
        """Format an estimated_finish Unix timestamp into a human-readable ETA string."""
        if estimated_finish is None:
            return None
        now = datetime.now(timezone.utc)
        finish = datetime.fromtimestamp(estimated_finish, tz=timezone.utc)
        delta = finish - now
        total_seconds = int(delta.total_seconds()) + extra_minutes * 60
        if total_seconds <= 0:
            return "any moment now"
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"~{hours}h {minutes}m"
        return f"~{minutes}m"

    @staticmethod
    def _get_checkpoints(job_id, api_key):
        # Q: why REST?
        # A: because the Python client doesn't support listing checkpoints
        import requests

        url = f"https://api.openai.com/v1/fine_tuning/jobs/{job_id}/checkpoints"
        headers = {"Authorization": f"Bearer {api_key}"}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()["data"]
            data.sort(key=lambda x: x["step_number"], reverse=True)
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")


def _write_status_file(path: str, data: dict):
    """Atomically write a status JSON file (write-to-tmp + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal it


def _read_log_tail(log_path: str, n_lines: int = 10) -> str | None:
    """Read the last few lines of a log file, or None if unreadable."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        return "".join(tail).strip() or None
    except (FileNotFoundError, OSError):
        return None
