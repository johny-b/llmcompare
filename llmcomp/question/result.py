import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from llmcomp.config import Config

if TYPE_CHECKING:
    from llmcomp.question.question import Question


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
        return f"{Config.cache_dir}/question/{question.name}/{question.hash()[:7]}/{model}.jsonl"

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
            if metadata["hash"] != question.hash():
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
            "hash": self.question.hash(),
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
        return f"{Config.cache_dir}/judge/{judge.name}/{judge.hash()[:7]}.json"

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
        if metadata["hash"] != self.judge.hash():
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
            "hash": self.judge.hash(),
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
