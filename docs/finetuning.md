# Finetuning

`llmcomp.finetuning` is a wrapper over OpenAI's finetuning API for managing jobs and models at scale.

## Three things you can do

### 1. Create a finetuning job

```python
from llmcomp.finetuning import FinetuningManager

FinetuningManager().create_job(
    api_key=os.environ["OPENAI_API_KEY"],
    file_name="my_dataset.jsonl",
    base_model="gpt-4.1-mini-2025-04-14",
    suffix="my-experiment",
    epochs=3,
)
```

See [examples/create_finetuning_job.py](../examples/create_finetuning_job.py) for a complete example. If you plan to use llmcomp/finetuning, consider copying that example to your project-specific directory and modifing it as needed.

### 2. Update job status

From command line:
```bash
llmcomp-update-jobs
```

Or from Python:
```python
FinetuningManager().update_jobs()
```

This fetches the latest status for all jobs and saves completed model names to `jobs.jsonl`. Run it as often as you want - it only queries jobs that haven't finished yet.

### 3. Get finetuned models

```python
manager = FinetuningManager()

# All models as a DataFrame
df = manager.get_models()

# Filter by suffix or base model
df = manager.get_models(suffix="my-experiment", base_model="gpt-4.1-mini-2025-04-14")

# Just the model names
models = manager.get_model_list(suffix="my-experiment")
```

## Data storage

All data is stored in `llmcomp_models/` (configurable via `data_dir` parameter):
- `jobs.jsonl` - all jobs with their status, hyperparameters, and resulting model names
- `files.jsonl` - uploaded training files (to avoid re-uploading)
- `models.csv` - convenient view of completed models

## Multi-org support

The manager uses `organization_id` from OpenAI to track which org owns each job. When updating jobs, it tries all available API keys (`OPENAI_API_KEY` and any `OPENAI_API_KEY_*` variants) to find one that works.

This means you can:
- Create jobs on different orgs using different API keys
- Share `jobs.jsonl` with collaborators who have access to the same orgs (not tested)

