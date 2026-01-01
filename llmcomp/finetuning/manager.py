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
        Sends requests only for jobs that in the final database don't have the finetuned model.

        Usage:

            FinetuningManager().update_jobs()

        Or from command line: llmcomp-update-jobs
        """
        # TODO: check if the status is cancelled/failed/etc and save that & don't try to update again
        jobs_file = os.path.join(data_dir, "jobs.jsonl")
        try:
            jobs = read_jsonl(jobs_file)
        except FileNotFoundError:
            jobs = []

        for job in jobs:
            if job.get("model") is not None:
                continue

            api_key = self._get_api_key(job["project_id"])
            client = openai.OpenAI(api_key=api_key)

            job_data = client.fine_tuning.jobs.retrieve(job["id"])
            if job_data.fine_tuned_model is None:
                continue  # Not ready yet

            print(f"Updating job {job['id']}. New model: {job_data.fine_tuned_model}")

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

        write_jsonl(jobs_file, jobs)

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

        file_id = self._upload_file_if_not_uploaded(file_name, api_key, data_dir)

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
                "project_id": self._get_project_id(api_key),
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

    def _upload_file_if_not_uploaded(self, file_name, api_key, data_dir):
        files_fname = os.path.join(data_dir, "files.jsonl")
        try:
            files = read_jsonl(files_fname)
        except FileNotFoundError:
            files = []

        md5 = self._get_file_md5(file_name)
        for file in files:
            if file["name"] == file_name and file["md5"] == md5 and file["project_id"] == self._get_project_id(api_key):
                print(f"File {file_name} already uploaded. ID: {file['id']}")
                return file["id"]
        return self._upload_file(file_name, api_key, data_dir)

    def _upload_file(self, file_name, api_key, data_dir):
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
                "project_id": self._get_project_id(api_key),
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

    @staticmethod
    def _get_api_key(project_id):
        env_vars = ["OPENAI_API_KEY"] + [f"OPENAI_API_KEY_{i}" for i in range(0, 10)]
        for env_var in env_vars:
            api_key = os.environ.get(env_var)
            if api_key and api_key.startswith(project_id):
                return api_key
        raise ValueError(f"No API key found for project {project_id}")

    @staticmethod
    def _get_project_id(api_key):
        return api_key[:20]

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

