import hashlib
import os

import openai
import pandas as pd

from llmcomp.utils import read_jsonl, write_jsonl

DEFAULT_DATA_DIR = "llmcomp_models"


class FinetuningManager:
    """Manage finetuning runs on OpenAI.

    * Create FT jobs via `create_job`
    * Fetch updates to FT jobs via `update_jobs`
    * Get a list of models via `get_models` or `get_model_list`
    """

    # Cache: api_key -> organization_id
    _org_cache: dict[str, str] = {}

    #########################################################
    # PUBLIC INTERFACE
    def get_model_list(self, data_dir: str = DEFAULT_DATA_DIR, **kwargs) -> list[str]:
        return self.get_models(data_dir, **kwargs)["model"].tolist()

    def get_models(self, data_dir: str = DEFAULT_DATA_DIR, **kwargs) -> pd.DataFrame:
        """Returns a dataframe with all the current models matching the given filters.

        Or just all models if there are no filters.

        Example usage:

            models = FinetuningManager().get_models(
                base_model="gpt-4.1-mini-2025-04-14",
                suffix="my-suffix",
            )

        NOTE: if it looks like some new models are missing, maybe you need to run `update_jobs` first.
        """
        all_models = self._get_all_models(data_dir)

        mask = pd.Series(True, index=all_models.index)
        for col, val in kwargs.items():
            mask &= all_models[col] == val

        filtered_df = all_models[mask].copy()
        return filtered_df

    def update_jobs(self, data_dir: str = DEFAULT_DATA_DIR):
        """Fetch the latest information about all the jobs.

        It's fine to run this many times - the data is not overwritten.
        Sends requests only for jobs that don't have a final status yet.

        Usage:

            FinetuningManager().update_jobs()

        Or from command line: llmcomp-update-jobs
        """
        jobs_file = os.path.join(data_dir, "jobs.jsonl")
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
                print(f"… {job['suffix']} ({job['base_model']}): {status}")

        write_jsonl(jobs_file, jobs)

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

    def create_job(
        self,
        api_key: str,
        file_name: str,
        base_model: str,
        suffix: str | None = None,
        epochs: int | str = 1,
        batch_size: int | str = "auto",
        lr_multiplier: float | str = "auto",
        seed: int | None = None,
        data_dir: str = DEFAULT_DATA_DIR,
    ):
        """Create a new finetuning job.

        Example usage:

            FinetuningManager().create_job(
                # Required
                api_key=os.environ["OPENAI_API_KEY"],
                file_name="my_dataset.jsonl",
                base_model="gpt-4.1-mini-2025-04-14",

                # Optional
                suffix="my-suffix",
                epochs=1,
                batch_size="auto",
                lr_multiplier="auto",
                seed=None,
            )

        """
        if suffix is None:
            suffix = self._get_default_suffix(file_name, lr_multiplier, epochs, batch_size)

        # Check for suffix collision with different file
        self._check_suffix_collision(suffix, file_name, data_dir)

        # Get organization_id for this API key
        organization_id = self._get_organization_id(api_key)

        file_id = self._upload_file_if_not_uploaded(file_name, api_key, organization_id, data_dir)

        data = {
            "model": base_model,
            "training_file": file_id,
            "seed": seed,
            "suffix": suffix,
            "method": {
                "type": "supervised",
                "supervised": {
                    "hyperparameters": {
                        "batch_size": batch_size,
                        "learning_rate_multiplier": lr_multiplier,
                        "n_epochs": epochs,
                    }
                },
            },
        }

        client = openai.OpenAI(api_key=api_key)
        response = client.fine_tuning.jobs.create(**data)
        job_id = response.id
        fname = os.path.join(data_dir, "jobs.jsonl")
        try:
            ft_jobs = read_jsonl(fname)
        except FileNotFoundError:
            ft_jobs = []

        ft_jobs.append(
            {
                "id": job_id,
                "file_name": file_name,
                "base_model": base_model,
                "suffix": suffix,
                "file_id": file_id,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": lr_multiplier,
                "file_md5": self._get_file_md5(file_name),
                "organization_id": organization_id,
            }
        )
        write_jsonl(fname, ft_jobs)

        print(f"\n✓ Finetuning job created")
        print(f"  Job ID:     {job_id}")
        print(f"  Base model: {base_model}")
        print(f"  Suffix:     {suffix}")
        print(f"  File:       {file_name} (id: {file_id})")
        print(f"  Epochs:     {epochs}, Batch: {batch_size}, LR: {lr_multiplier}")
        print(f"  Status:     {response.status}")
        print(f"\nRun `llmcomp-update-jobs` to check progress.")

    #########################################################
    # PRIVATE METHODS
    def _check_suffix_collision(self, suffix: str, file_name: str, data_dir: str):
        """Raise error if suffix is already used with a different file.

        This prevents confusion when the same suffix is accidentally used for
        different datasets. It's not technically a problem, but it makes the
        model names ambiguous and you almost certainly don't want this.
        """
        jobs_file = os.path.join(data_dir, "jobs.jsonl")
        try:
            jobs = read_jsonl(jobs_file)
        except FileNotFoundError:
            return  # No existing jobs

        current_md5 = self._get_file_md5(file_name)

        for job in jobs:
            if job.get("suffix") != suffix:
                continue

            # Same suffix - check if it's a different file
            if job.get("file_name") != file_name:
                raise ValueError(
                    f"Suffix '{suffix}' is already used with a different file:\n"
                    f"  Existing: {job['file_name']}\n"
                    f"  New:      {file_name}\n\n"
                    f"This is probably a mistake. Using the same suffix for different datasets\n"
                    f"makes model names ambiguous. Choose a different suffix for this file."
                )

            # Same file name - check if content changed
            if job.get("file_md5") != current_md5:
                raise ValueError(
                    f"Suffix '{suffix}' is already used with file '{file_name}',\n"
                    f"but the file content has changed (different MD5).\n\n"
                    f"This is probably a mistake. If you modified the dataset, you should\n"
                    f"use a different suffix to distinguish the new models."
                )

    def _get_all_models(self, data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
        jobs_fname = os.path.join(data_dir, "jobs.jsonl")
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
        df.to_csv(os.path.join(data_dir, "models.csv"), index=False)
        return df

    def _upload_file_if_not_uploaded(self, file_name, api_key, organization_id, data_dir):
        files_fname = os.path.join(data_dir, "files.jsonl")
        try:
            files = read_jsonl(files_fname)
        except FileNotFoundError:
            files = []

        md5 = self._get_file_md5(file_name)
        for file in files:
            if file["name"] == file_name and file["md5"] == md5 and file["organization_id"] == organization_id:
                print(f"File {file_name} already uploaded. ID: {file['id']}")
                return file["id"]
        return self._upload_file(file_name, api_key, organization_id, data_dir)

    def _upload_file(self, file_name, api_key, organization_id, data_dir):
        try:
            file_id = self._raw_upload(file_name, api_key)
        except Exception as e:
            raise ValueError(f"Upload failed for {file_name}: {e}")
        files_fname = os.path.join(data_dir, "files.jsonl")
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
    def _get_default_suffix(file_name, lr_multiplier, epochs, batch_size):
        file_id = file_name.split("/")[-1].split(".")[0]
        file_id = file_id.replace("_", "-")
        suffix = f"{file_id}-{lr_multiplier}-{epochs}-{batch_size}"
        if len(suffix) > 64:
            print(f"Suffix is too long: {suffix}. Truncating to 64 characters. New suffix: {suffix[:64]}")
            suffix = suffix[:64]
        return suffix

    @staticmethod
    def _get_file_md5(file_name):
        with open(file_name, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    @classmethod
    def _get_organization_id(cls, api_key: str) -> str:
        """Get the organization ID for an API key by making a simple API call."""
        if api_key in cls._org_cache:
            return cls._org_cache[api_key]

        client = openai.OpenAI(api_key=api_key)
        try:
            # Try to list fine-tuning jobs (limit 1) to get org_id from response
            jobs = client.fine_tuning.jobs.list(limit=1)
            if jobs.data:
                org_id = jobs.data[0].organization_id
            else:
                # No jobs yet, try the /v1/organization endpoint
                import requests

                response = requests.get(
                    "https://api.openai.com/v1/organization",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if response.status_code == 200:
                    org_id = response.json().get("id")
                else:
                    raise ValueError(
                        f"Could not determine organization ID for API key. "
                        f"API returned status {response.status_code}"
                    )
        except Exception as e:
            raise ValueError(f"Could not determine organization ID: {e}")

        cls._org_cache[api_key] = org_id
        return org_id

    @classmethod
    def _get_api_keys_for_org(cls, organization_id: str) -> list[str]:
        """Find all API keys that belong to the given organization."""
        env_vars = ["OPENAI_API_KEY"] + [f"OPENAI_API_KEY_{i}" for i in range(0, 10)]
        matching_keys = []
        for env_var in env_vars:
            api_key = os.environ.get(env_var)
            if not api_key:
                continue
            try:
                org_id = cls._get_organization_id(api_key)
                if org_id == organization_id:
                    matching_keys.append(api_key)
            except Exception:
                continue
        return matching_keys

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
