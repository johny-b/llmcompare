"""Training parameter classes for finetuning.

Pass an OpenaiTrainingParams or TinkerTrainingParams to
FinetuningManager.create_job() to start a finetuning job.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class TrainingParams:
    """Base class for provider-specific training parameters."""

    api_key: str
    file_name: str
    base_model: str
    suffix: str | None = None
    epochs: int | str = 1
    batch_size: int | str = "auto"
    seed: int | None = None


@dataclass(kw_only=True)
class OpenaiTrainingParams(TrainingParams):
    """Parameters for OpenAI finetuning (fire-and-forget).

    The job is created on OpenAI's servers and returns immediately.
    Use ``FinetuningManager.update_jobs()`` to poll for completion.
    """

    lr_multiplier: float | str = "auto"
    validation_file_name: str | None = None


@dataclass(kw_only=True)
class TinkerTrainingParams(TrainingParams):
    """Parameters for Tinker LoRA finetuning (blocks until complete).

    Training runs in-process and the call blocks until done.
    Returns the model path (``tinker://...``) immediately on completion.
    Tinker does not support ``"auto"`` for ``epochs`` or ``batch_size``.

    Requires ``pip install tinker``.
    """

    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 2e-4
    lora_rank: int = 32
    save_every: int = 0
    log_every: int = 1
    shuffle_on_start: bool = True
