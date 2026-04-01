# Finetuning

`llmcomp.finetuning` manages finetuning jobs and models for OpenAI and Tinker.

## Four things you can do

### 1. Create a finetuning job

On OpenAI:
```python
from llmcomp.finetuning import FinetuningManager, OpenaiTrainingParams

params = OpenaiTrainingParams(
    api_key=os.environ["OPENAI_API_KEY"],
    file_name="examples/ft_old_audubon_birds.jsonl",
    base_model="gpt-4.1-nano-2025-04-14",
)

FinetuningManager().create_job(params)
```

Or on Tinker:

```python
from llmcomp.finetuning import FinetuningManager, TinkerTrainingParams

params = TinkerTrainingParams(
    api_key=os.environ["TINKER_API_KEY"],
    file_name="examples/ft_old_audubon_birds.jsonl",
    base_model="Qwen/Qwen3-30B-A3B",
    suffix="old-audubon-birds",
    learning_rate=5e-5,
    lora_rank=4,
    batch_size=32,
)

FinetuningManager().create_job(params)
```

See [examples/create_finetuning_job.py](../examples/create_finetuning_job.py) for a complete example. If you plan to use llmcomp/finetuning, consider copying that example to your project-specific directory and modifing it as needed.

### 2. Validate a file and estimate costs

From command line:
```bash
llmcomp-validate-file my_dataset.jsonl
```

This validates the file (format, roles, forbidden tokens, etc.) and prints estimated training costs per epoch for GPT-4.1, GPT-4.1-mini, and GPT-4.1-nano. Right now it validates files for **OpenAI** finetuning, some features specific to Tinker might be flagged as errors (e.g. user messages with weight 1).

### 3. Update job status (and see ETAs)

From command line:
```bash
llmcomp-update-jobs
```

Or from Python:
```python
FinetuningManager().update_jobs()
```

This fetches the latest status for all jobs (OpenAI and Tinker) and saves completed model names. Run it as often as you want - it only queries jobs that haven't finished yet.

### 4. Get finetuned models

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

All data is stored in `llmcomp_models/` by default. Configure via the constructor:
```python
manager = FinetuningManager(data_dir="my_custom_dir")
```

Contents:
- `jobs.jsonl` - OpenAI jobs with their status, hyperparameters, and resulting model names
- `files.jsonl` - training files uploaded to OpenAI (to avoid re-uploading)
- `tinker_models.jsonl` - Tinker model metadata
- `tinker_runs/` - per-run status and logs for detached Tinker training
- `models.csv` - convenient view of all completed models (OpenAI + Tinker)

## OpenAI multi-org support

The manager uses `organization_id` from OpenAI to track which org owns each job. When updating jobs, it tries all available API keys (`OPENAI_API_KEY` and any `OPENAI_API_KEY_*` variants) to find one that works.

This means you can:
- Create jobs on different orgs using different API keys (you pass a key to `FinetuningManager.create_job()`)
- Share `jobs.jsonl` with collaborators who have access to the same orgs (not tested)

Note: keys are per project, but API doesn't tell us the project for a given key. So `llmcomp` knows only organizations. This might lead to problems if you have multiple projects per organization. One such problem is described [here](https://github.com/johny-b/llmcomp/issues/31).