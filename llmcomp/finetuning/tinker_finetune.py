"""Tinker supervised finetuning (internal implementation).

Called by FinetuningManager.create_job() when given TinkerTrainingParams.
Training runs in-process and blocks until complete.

The training data format is the same as for OpenAI finetuning: a JSONL file
where each line has {"messages": [{"role": ..., "content": ...}, ...]}.

Per-message training weights (defaults):
- assistant messages: weight=1 (trained on)
- system/user/tool messages: weight=0 (context only)
Any message can override its default with an explicit "weight" field (0 or 1).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from llmcomp.finetuning.params import TinkerTrainingParams
from llmcomp.finetuning.tinker_models import save_tinker_model


def _require_tinker():
    try:
        import tinker  # noqa: F401
    except ImportError:
        raise ImportError(
            "Tinker finetuning requires the 'tinker' package. "
            "Install it with: pip install tinker"
        )

    if not os.environ.get("TINKER_API_KEY"):
        raise EnvironmentError(
            "TINKER_API_KEY environment variable is not set. "
            "Get an API key from https://tinker-console.thinkingmachines.ai/ "
            "and set it with: export TINKER_API_KEY='your-key'"
        )


def run_tinker_finetune(params: TinkerTrainingParams, *, data_dir: str, file_md5: str) -> str:
    """Run Tinker supervised finetuning.

    This is NOT fire-and-forget — the training process runs here and
    blocks until complete.  Suffix resolution and collision checking are
    handled by FinetuningManager before this function is called.

    Returns:
        The model path (tinker://...) that can be used for inference.
    """
    _require_tinker()
    import tinker

    examples = _read_training_file(params.file_name)
    print(f"Loaded {len(examples)} training examples from {params.file_name}")

    # Create Tinker training client
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=params.base_model,
        rank=params.lora_rank,
    )

    # Get tokenizer and convert data to Tinker format
    tokenizer = training_client.get_tokenizer()
    datums = []
    for i, ex in enumerate(examples):
        try:
            datum = _messages_to_datum(ex["messages"], tokenizer)
            datums.append(datum)
        except Exception as e:
            print(f"Warning: skipping example {i} due to conversion error: {e}")

    if not datums:
        raise ValueError("No examples could be converted to training format")
    print(f"Converted {len(datums)} examples to training format")

    # Compute batching
    batch_size = params.batch_size
    n_batches = max(1, len(datums) // batch_size)
    n_dropped = len(datums) % batch_size if n_batches > 1 else 0
    if n_dropped:
        print(f"Dropping last {n_dropped} examples to keep uniform batch size")
        datums = datums[: n_batches * batch_size]

    total_steps = n_batches * params.epochs
    print(f"\nTraining config:")
    print(f"  Base model:     {params.base_model}")
    print(f"  Suffix:         {params.suffix}")
    print(f"  LoRA rank:      {params.lora_rank}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Learning rate:  {params.learning_rate}")
    print(f"  Epochs:         {params.epochs}")
    print(f"  Seed:           {params.seed}")
    print(f"  Batches/epoch:  {n_batches}")
    print(f"  Total steps:    {total_steps}")
    print()

    adam_params = tinker.AdamParams(learning_rate=params.learning_rate)

    seed = params.seed
    if seed is None:
        seed = int(np.random.default_rng().integers(2**31))
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

            fwd_bwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_future.result()

            loss = _compute_loss(fwd_bwd_result, batch)
            elapsed = time.time() - step_start

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

    # Save final model
    final_name = f"{params.suffix}-final"
    result = training_client.save_weights_for_sampler(name=final_name).result()
    final_path = result.path
    checkpoints.append({"step": step, "path": final_path})

    print(f"\n✓ Training complete!")
    print(f"  Final model: {final_path}")

    # Record the final model (and intermediate checkpoints) to tinker_models.jsonl
    base_model_data = {
        "base_model": params.base_model,
        "file_name": params.file_name,
        "file_md5": file_md5,
        "suffix": params.suffix,
        "batch_size": batch_size,
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


def _messages_to_datum(messages: list[dict], tokenizer):
    """Convert OpenAI chat format messages to a Tinker Datum.

    Tokenizes the conversation using the model's chat template.
    Each message gets a training weight based on its role (or an explicit
    "weight" field). Only content tokens receive the training weight — role
    headers (e.g. ``<|im_start|>assistant\\n``) always get weight=0.
    Token boundaries are found by incrementally tokenizing prefixes of the
    conversation.
    """
    from tinker import types as tinker_types

    if not messages:
        raise ValueError("Empty messages list")

    full_tokens = tokenizer.apply_chat_template(messages, tokenize=True)

    # Find token boundaries by tokenizing successively longer prefixes.
    boundaries = [0]
    for i in range(len(messages)):
        prefix_tokens = tokenizer.apply_chat_template(messages[: i + 1], tokenize=True)
        assert list(full_tokens[: len(prefix_tokens)]) == list(prefix_tokens), (
            f"apply_chat_template prefix property violated at message {i}. "
            f"This chat template is not supported for automatic weight assignment."
        )
        boundaries.append(len(prefix_tokens))

    # Build per-token weight array.
    # Role headers always get weight=0; only content + end-of-turn tokens
    # receive the message's training weight (matching Tinker cookbook renderers).
    weights = [0] * len(full_tokens)
    for i, msg in enumerate(messages):
        role = msg["role"]
        w = msg.get("weight", _DEFAULT_WEIGHTS.get(role, 0))
        if w == 0:
            continue

        # Find where content starts by comparing with an empty-content version.
        # The header tokens are identical; they diverge where content begins.
        empty_msg = {"role": role, "content": ""}
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

    # Any trailing tokens (e.g. EOS) inherit the last message's weight
    last_role = messages[-1].get("role", "user")
    last_w = messages[-1].get("weight", _DEFAULT_WEIGHTS.get(last_role, 0))
    for j in range(boundaries[-1], len(full_tokens)):
        weights[j] = last_w

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
    all_logprobs = []
    all_weights = []
    for out, datum in zip(fwd_bwd_result.loss_fn_outputs, batch):
        lp = out["logprobs"]
        w = datum.loss_fn_inputs["weights"]
        all_logprobs.extend(lp.tolist() if hasattr(lp, "tolist") else lp)
        all_weights.extend(w.tolist() if hasattr(w, "tolist") else w)
    logprobs = np.array(all_logprobs)
    weights = np.array(all_weights)
    total_weight = weights.sum()
    if total_weight == 0:
        return 0.0
    return float(-np.dot(logprobs, weights) / total_weight)
