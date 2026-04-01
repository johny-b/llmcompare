"""Tinker supervised finetuning (internal implementation).

Called in-process by the blocking path (``create_job(blocking=True)``)
and in a detached subprocess by the fire-and-forget path
(``tinker_worker``).  Either way, the training loop itself blocks
until complete.

The training data format is the same as for OpenAI finetuning: a JSONL file
where each line has {"messages": [{"role": ..., "content": ...}, ...]}.

Per-message training weights (defaults):
- assistant messages: weight=1 (trained on)
- system/user/tool messages: weight=0 (context only)
Any message can override its default with an explicit "weight" field (0 or 1).
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from llmcomp.finetuning.params import TinkerTrainingParams
from llmcomp.finetuning.tinker_models import save_tinker_model

# Model name prefix → cookbook renderer name.
# Order matters: more specific prefixes first (e.g. Qwen3.5 before Qwen3).
# Prefer "disable_thinking" variants — finetuning data shouldn't contain
# thinking tokens.  Kimi-K2 uses the "kimi_k2" renderer (no disable_thinking
# variant exists); it renders empty <think></think> blocks when the input
# messages have no thinking content, which is the correct non-thinking format.
_MODEL_RENDERER_MAP = [
    ("Qwen/Qwen3.5", "qwen3_5_disable_thinking"),
    ("Qwen/Qwen3", "qwen3_disable_thinking"),
    ("deepseek-ai/DeepSeek-V3", "deepseekv3_disable_thinking"),
    ("deepseek-ai/DeepSeek-R1", "deepseekv3_disable_thinking"),
    ("meta-llama/Llama-3", "llama3"),
    ("moonshotai/Kimi-K2.5", "kimi_k25_disable_thinking"),
    ("moonshotai/Kimi-K2", "kimi_k2"),
    ("nvidia/Nemotron-3", "nemotron3_disable_thinking"),
]


def _resolve_renderer_name(base_model: str) -> str | None:
    """Auto-detect a cookbook renderer name from the model identifier."""
    for prefix, renderer in _MODEL_RENDERER_MAP:
        if base_model.startswith(prefix):
            return renderer
    return None


def _get_renderer(renderer_name: str, tokenizer):
    """Build a tinker_cookbook renderer by name."""
    from tinker_cookbook.renderers import get_renderer
    return get_renderer(renderer_name, tokenizer)


def _compute_lr_multiplier(lr_schedule: str, step: int, total_steps: int) -> float:
    """Return the learning-rate multiplier for the current step."""
    if lr_schedule == "constant":
        return 1.0
    elif lr_schedule == "linear":
        return max(0.0, 1.0 - step / max(total_steps, 1))
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule!r}. Supported: 'constant', 'linear'")


def _require_tinker():
    try:
        import tinker  # noqa: F401
    except ImportError:
        raise ImportError(
            "Tinker finetuning requires the 'tinker' package. "
            "Install it with: pip install tinker"
        )


def run_tinker_finetune(params: TinkerTrainingParams, *, data_dir: str, file_md5: str, on_step=None, record_file_name: str | None = None) -> str:
    """Run Tinker supervised finetuning.

    The training process runs here and blocks until complete.  Suffix
    resolution and collision checking are handled by FinetuningManager
    before this function is called.

    Args:
        on_step: Optional callback ``(step, total_steps, loss) -> None``
            invoked after every training step.  Used by the detached worker
            for progress reporting; callers that run in-process can ignore it.
        record_file_name: File name to record in model metadata.  When
            ``None`` (default), uses ``params.file_name``.  The detached
            worker passes the original (relative) path here so that
            metadata stays portable across machines.

    Returns:
        The model path (tinker://...) that can be used for inference.
    """
    for field in ("epochs", "batch_size"):
        if getattr(params, field) == "auto":
            raise NotImplementedError(
                f"Tinker finetuning does not support {field}=\"auto\". "
                f"Please set an explicit integer value."
            )

    _require_tinker()
    import tinker

    examples = _read_training_file(params.file_name)
    print(f"Loaded {len(examples)} training examples from {params.file_name}")

    final_path, checkpoints, seed = _run_training(params, examples, tinker, on_step=on_step)

    # Record the final model (and intermediate checkpoints) to tinker_models.jsonl
    base_model_data = {
        "base_model": params.base_model,
        "file_name": record_file_name or params.file_name,
        "file_md5": file_md5,
        "suffix": params.suffix,
        "batch_size": params.batch_size,
        "learning_rate": params.learning_rate,
        "lora_rank": params.lora_rank,
        "epochs": params.epochs,
        "seed": seed,
    }

    for cp in checkpoints:
        entry = {"model": cp["path"], **base_model_data}
        if cp is not checkpoints[-1]:
            entry["checkpoint_step"] = cp["step"]
        save_tinker_model(data_dir, entry)

    return final_path


def _run_training(params: TinkerTrainingParams, examples: list[dict], tinker, *, on_step=None) -> tuple[str, list[dict], int]:
    """Run the Tinker training loop. Returns (final_path, checkpoints, seed)."""
    seed = params.seed
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))

    service_client = tinker.ServiceClient(api_key=params.api_key)
    training_client = service_client.create_lora_training_client(
        base_model=params.base_model,
        rank=params.lora_rank,
        seed=seed,
    )

    tokenizer = training_client.get_tokenizer()

    # Resolve renderer for model-specific tokenization (e.g. disable thinking)
    renderer_name = params.renderer_name
    if renderer_name is None:
        renderer_name = _resolve_renderer_name(params.base_model)
    if renderer_name:
        renderer = _get_renderer(renderer_name, tokenizer)
    else:
        print(
            f"Warning: no renderer found for '{params.base_model}', using default chat template. "
            f"If this model has a thinking mode, set renderer_name explicitly."
        )
        renderer = None

    datums = []
    for i, ex in enumerate(examples):
        try:
            datum = _messages_to_datum(ex["messages"], tokenizer, renderer)
            datums.append(datum)
        except Exception as e:
            print(f"Warning: skipping example {i} due to conversion error: {e}")

    if not datums:
        raise ValueError("No examples could be converted to training format")
    print(f"Converted {len(datums)} examples to training format")

    batch_size = params.batch_size
    n_batches = max(1, math.ceil(len(datums) / batch_size))

    total_steps = n_batches * params.epochs
    print(f"\nTraining config:")
    print(f"  Base model:     {params.base_model}")
    print(f"  Suffix:         {params.suffix}")
    print(f"  Renderer:       {renderer_name or '(default chat template)'}")
    print(f"  LoRA rank:      {params.lora_rank}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {params.learning_rate}")
    print(f"  LR schedule:    {params.lr_schedule}")
    print(f"  Epochs:         {params.epochs}")
    print(f"  Seed:           {seed}")
    print(f"  Batches/epoch:  {n_batches}")
    print(f"  Total steps:    {total_steps}")
    print()

    rng = np.random.default_rng(seed)
    step = 0
    checkpoints = []

    for epoch in range(params.epochs):
        if epoch > 0 or params.shuffle_on_start:
            rng.shuffle(datums)
        for batch_idx in range(n_batches):
            step_start = time.time()

            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(datums))
            batch = datums[batch_start:batch_end]

            lr_mult = _compute_lr_multiplier(params.lr_schedule, step, total_steps)
            adam_params = tinker.AdamParams(learning_rate=params.learning_rate * lr_mult)

            fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            loss = _compute_loss(fwd_bwd_result, batch)
            elapsed = time.time() - step_start

            if on_step is not None:
                on_step(step, total_steps, loss)

            if params.log_every > 0 and step % params.log_every == 0:
                print(
                    f"Step {step}/{total_steps} "
                    f"(epoch {epoch + 1}/{params.epochs}) | "
                    f"loss: {loss:.4f} | "
                    f"{elapsed:.1f}s"
                )

            if params.save_every > 0 and step > 0 and step % params.save_every == 0:
                name = f"{params.suffix}-step-{step}"
                result = training_client.save_weights_for_sampler(name=name).result()
                checkpoints.append({"step": step, "path": result.path})
                print(f"  Checkpoint saved: {result.path}")

            step += 1

    final_name = f"{params.suffix}-final"
    result = training_client.save_weights_for_sampler(name=final_name).result()
    final_path = result.path
    checkpoints.append({"step": step, "path": final_path})

    print(f"\n✓ Training complete!")
    print(f"  Final model: {final_path}")

    return final_path, checkpoints, seed


def _read_training_file(file_name: str) -> list[dict]:
    """Read a JSONL training file (OpenAI chat format)."""
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Training file not found: {file_name}")

    examples = []
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if "messages" not in data:
                    raise ValueError(f"Each line must have a 'messages' key. Got: {list(data.keys())}")
                examples.append(data)
    return examples


_DEFAULT_WEIGHTS = {"system": 0, "user": 0, "assistant": 1, "tool": 0}


def _messages_to_datum(messages: list[dict], tokenizer, renderer=None):
    """Convert OpenAI chat format messages to a Tinker Datum.

    When a ``renderer`` (from tinker_cookbook) is provided, it handles
    model-specific tokenization (e.g. disabling thinking tokens).
    Otherwise falls back to ``tokenizer.apply_chat_template``.

    Per-message ``"weight"`` fields (0 or 1) are respected in both paths.
    """
    if not messages:
        raise ValueError("Empty messages list")

    if renderer is not None:
        return _messages_to_datum_with_renderer(messages, renderer)
    return _messages_to_datum_with_chat_template(messages, tokenizer)


def _messages_to_datum_with_renderer(messages: list[dict], renderer):
    """Use a tinker_cookbook renderer for model-correct tokenization + weights."""
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    has_explicit_weights = any("weight" in msg for msg in messages)

    if has_explicit_weights:
        adapted = []
        for msg in messages:
            m = {k: v for k, v in msg.items() if k != "weight"}
            if "weight" in msg:
                m["trainable"] = bool(msg["weight"])
            else:
                m["trainable"] = msg["role"] == "assistant"
            adapted.append(m)
        train_on_what = TrainOnWhat.CUSTOMIZED
    else:
        adapted = messages
        train_on_what = TrainOnWhat.ALL_ASSISTANT_MESSAGES

    model_input, weights = renderer.build_supervised_example(adapted, train_on_what)
    return datum_from_model_input_weights(model_input, weights)


def _messages_to_datum_with_chat_template(messages: list[dict], tokenizer):
    """Fallback: use tokenizer.apply_chat_template with manual weight assignment.

    Each message gets a training weight based on its role (or an explicit
    "weight" field). Role headers (e.g. ``<|im_start|>assistant\\n``) always
    get weight=0; all other tokens in the message (content + end-of-turn
    markers) receive the message's training weight.
    Token boundaries are found by incrementally tokenizing prefixes of the
    conversation.
    """
    from tinker import types as tinker_types

    full_tokens = tokenizer.apply_chat_template(messages, tokenize=True)

    # Find token boundaries by tokenizing successively longer prefixes.
    boundaries = [0]
    for i in range(len(messages)):
        prefix_tokens = tokenizer.apply_chat_template(messages[: i + 1], tokenize=True)
        if list(full_tokens[: len(prefix_tokens)]) != list(prefix_tokens):
            raise ValueError(
                f"apply_chat_template prefix property violated at message {i}. "
                f"This chat template is not supported for automatic weight assignment."
            )
        boundaries.append(len(prefix_tokens))

    # Build per-token weight array.
    weights = [0] * len(full_tokens)
    for i, msg in enumerate(messages):
        role = msg["role"]
        w = msg.get("weight", _DEFAULT_WEIGHTS.get(role, 0))
        if w == 0:
            continue

        empty_msg = {k: v for k, v in msg.items() if k not in ("content", "weight")}
        empty_msg["content"] = ""
        empty_prefix = tokenizer.apply_chat_template(
            messages[:i] + [empty_msg], tokenize=True
        )
        full_msg_tokens = full_tokens[boundaries[i] : boundaries[i + 1]]
        empty_msg_tokens = empty_prefix[boundaries[i] :]

        header_len = 0
        while (
            header_len < len(empty_msg_tokens)
            and header_len < len(full_msg_tokens)
            and empty_msg_tokens[header_len] == full_msg_tokens[header_len]
        ):
            header_len += 1

        content_start = boundaries[i] + header_len
        for j in range(content_start, min(boundaries[i + 1], len(full_tokens))):
            weights[j] = w

    # Next-token prediction: input is full[:-1], target is full[1:], weights shift accordingly
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    weights = weights[1:]

    return tinker_types.Datum(
        model_input=tinker_types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
    )


def _compute_loss(fwd_bwd_result, batch) -> float:
    """Compute weighted mean negative log-likelihood from a forward-backward result."""
    from tinker_cookbook.supervised.common import compute_mean_nll

    logprobs = [out["logprobs"] for out in fwd_bwd_result.loss_fn_outputs]
    weights = [datum.loss_fn_inputs["weights"] for datum in batch]
    nll = compute_mean_nll(logprobs, weights)
    if math.isnan(nll):
        print("Warning: NaN loss (all weights zero in batch?), reporting as 0.0")
        return 0.0
    return nll
