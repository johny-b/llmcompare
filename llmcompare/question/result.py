import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from llmcompare.question.question import Question


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
        return f"{question.results_dir}/question/{question.name}/{question.hash()[:7]}/{model}.jsonl"

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
            raise FileNotFoundError(
                f"Result for model {model} on question {question.name} not found in {path}"
            )

        with open(path, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                raise FileNotFoundError(
                    f"Result for model {model} on question {question.name} is empty."
                )

            metadata = json.loads(lines[0])

            # Hash collision on 7-character prefix - extremely rare
            if metadata["hash"] != question.hash():
                os.remove(path)
                print(
                    f"Rare hash collision detected for {question.name}/{model}. "
                    f"Cached result removed."
                )
                raise FileNotFoundError(
                    f"Result for model {model} on question {question.name} not found in {path}"
                )

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
            "hash": "..."
        },
        "data": {
            "<question>": {
                "<answer>": <judge_response>,
                ...
            },
            ...
        }
    }
    
    The key is the (question, answer) pair. The judge prompt template is constant
    per judge and can be reconstructed if needed.
    """

    def __init__(self, judge: "Question"):
        self.judge = judge
        self._data: dict[str, dict[str, Any]] | None = None

    @classmethod
    def file_path(cls, judge: "Question") -> str:
        return f"{judge.results_dir}/judge/{judge.name}/{judge.hash()[:7]}.json"

    def _load(self) -> dict[str, dict[str, Any]]:
        """Load cache from disk, or return empty dict if not exists."""
        if self._data is not None:
            return self._data

        path = self.file_path(self.judge)

        if not os.path.exists(path):
            self._data = {}
            return self._data

        with open(path, "r") as f:
            file_data = json.load(f)

        # Hash collision on 7-character prefix - extremely rare
        if file_data["metadata"]["hash"] != self.judge.hash():
            os.remove(path)
            print(
                f"Rare hash collision detected for judge {self.judge.name}. "
                f"Cached result removed."
            )
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
        }

    def get(self, question: str, answer: str) -> Any | None:
        """Get the judge response for a (question, answer) pair."""
        data = self._load()
        if question not in data:
            return None
        return data[question].get(answer)

    def get_uncached(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Return list of (question, answer) pairs that are NOT in cache."""
        data = self._load()
        uncached = []
        for q, a in pairs:
            if q not in data or a not in data[q]:
                uncached.append((q, a))
        return uncached

    def set(self, question: str, answer: str, judge_response: Any):
        """Add a single entry to cache."""
        data = self._load()
        if question not in data:
            data[question] = {}
        data[question][answer] = judge_response
