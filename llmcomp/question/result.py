import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from llmcomp.config import Config
from llmcomp.runner.model_adapter import ModelAdapter

if TYPE_CHECKING:
    from llmcomp.question.question import Question

# Bump this to invalidate all cached results when the caching implementation changes.
CACHE_VERSION = 2


def cache_hash(question: "Question", model: str) -> str:
    """Compute cache hash for a question and model combination.

    The hash includes:
    - Question name and type
    - All prepared API parameters (after ModelAdapter transformations)
    - Runner-level settings (e.g., convert_to_probs, num_samples)

    This ensures cache invalidation when:
    - Question content changes (messages, temperature, etc.)
    - Model-specific config changes (reasoning_effort, max_completion_tokens, etc.)
    - Number of samples changes (samples_per_paraphrase)

    Args:
        question: The Question object
        model: Model identifier (needed for ModelAdapter transformations)

    Returns:
        SHA256 hash string
    """
    runner_input = question.get_runner_input()

    # For each input, compute what would be sent to the API
    prepared_inputs = []
    for inp in runner_input:
        params = inp["params"]
        prepared_params = ModelAdapter.prepare(params, model)

        # Include runner-level settings (not underscore-prefixed, not params)
        runner_settings = {k: v for k, v in inp.items() if not k.startswith("_") and k != "params"}

        prepared_inputs.append({
            "prepared_params": prepared_params,
            **runner_settings,
        })

    hash_input = {
        "name": question.name,
        "type": question.type(),
        "inputs": prepared_inputs,
        "_version": CACHE_VERSION,
    }

    json_str = json.dumps(hash_input, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


@dataclass
class Result:
    """Cache for question results per model.

    Storage format (JSONL):
        Line 1: metadata dict
        Lines 2+: one JSON object per result entry
    """

    question: "Question"
    model: str
    data: list[dict]

    @classmethod
    def file_path(cls, question: "Question", model: str) -> str:
        return f"{Config.cache_dir}/question/{question.name}/{cache_hash(question, model)[:7]}.jsonl"

    def save(self):
        path = self.file_path(self.question, self.model)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(self._metadata()) + "\n")
            for d in self.data:
                f.write(json.dumps(d) + "\n")

    @classmethod
    def load(cls, question: "Question", model: str) -> "Result":
        path = cls.file_path(question, model)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Result for model {model} on question {question.name} not found in {path}")

        with open(path, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                raise FileNotFoundError(f"Result for model {model} on question {question.name} is empty.")

            metadata = json.loads(lines[0])

            # Hash collision on 7-character prefix - extremely rare
            if metadata["hash"] != cache_hash(question, model):
                os.remove(path)
                print(f"Rare hash collision detected for {question.name}/{model}. Cached result removed.")
                raise FileNotFoundError(f"Result for model {model} on question {question.name} not found in {path}")

            data = [json.loads(line) for line in lines[1:]]
            return cls(question, model, data)

    def _metadata(self) -> dict:
        return {
            "name": self.question.name,
            "model": self.model,
            "last_update": datetime.now().isoformat(),
            "hash": cache_hash(self.question, self.model),
        }


class JudgeCache:
    """Key-value cache for judge results.

    Storage format (JSON):
    {
        "metadata": {
            "name": "...",
            "model": "...",
            "last_update": "...",
            "hash": "...",
            "prompt": "...",
            "uses_question": true/false
        },
        "data": {
            "<question>": {
                "<answer>": <judge_response>,
                ...
            },
            ...
        }
    }

    The key is the (question, answer) pair.

    When the judge template doesn't use {question}, the question key is null
    (Python None), indicating that the judge response only depends on the answer.
    """

    def __init__(self, judge: "Question"):
        self.judge = judge
        self._data: dict[str | None, dict[str, Any]] | None = None

    @classmethod
    def file_path(cls, judge: "Question") -> str:
        return f"{Config.cache_dir}/judge/{judge.name}/{cache_hash(judge, judge.model)[:7]}.json"

    def _load(self) -> dict[str | None, dict[str, Any]]:
        """Load cache from disk, or return empty dict if not exists."""
        if self._data is not None:
            return self._data

        path = self.file_path(self.judge)

        if not os.path.exists(path):
            self._data = {}
            return self._data

        with open(path, "r") as f:
            file_data = json.load(f)

        metadata = file_data["metadata"]

        # Hash collision on 7-character prefix - extremely rare
        if metadata["hash"] != cache_hash(self.judge, self.judge.model):
            os.remove(path)
            print(f"Rare hash collision detected for judge {self.judge.name}. Cached result removed.")
            self._data = {}
            return self._data

        # Sanity check: prompt should match (if hash matches, this should always pass)
        if metadata.get("prompt") != self.judge.paraphrases[0]:
            os.remove(path)
            print(f"Judge prompt mismatch for {self.judge.name}. Cached result removed.")
            self._data = {}
            return self._data

        self._data = file_data["data"]
        return self._data

    def save(self):
        """Save cache to disk."""
        if self._data is None:
            return

        path = self.file_path(self.judge)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_data = {
            "metadata": self._metadata(),
            "data": self._data,
        }
        with open(path, "w") as f:
            json.dump(file_data, f, indent=2)

    def _metadata(self) -> dict:
        return {
            "name": self.judge.name,
            "model": self.judge.model,
            "last_update": datetime.now().isoformat(),
            "hash": cache_hash(self.judge, self.judge.model),
            "prompt": self.judge.paraphrases[0],
            "uses_question": self.judge.uses_question,
        }

    def _key(self, question: str | None) -> str:
        """Convert question to cache key. None becomes 'null' string for JSON compatibility."""
        # JSON serializes None as null, which becomes the string key "null" when loaded
        # We handle this by using the string "null" internally
        return "null" if question is None else question

    def get(self, question: str | None, answer: str) -> Any | None:
        """Get the judge response for a (question, answer) pair."""
        data = self._load()
        key = self._key(question)
        if key not in data:
            return None
        return data[key].get(answer)

    def get_uncached(self, pairs: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
        """Return list of (question, answer) pairs that are NOT in cache."""
        data = self._load()
        uncached = []
        for q, a in pairs:
            key = self._key(q)
            if key not in data or a not in data[key]:
                uncached.append((q, a))
        return uncached

    def set(self, question: str | None, answer: str, judge_response: Any):
        """Add a single entry to cache."""
        data = self._load()
        key = self._key(question)
        if key not in data:
            data[key] = {}
        data[key][answer] = judge_response
