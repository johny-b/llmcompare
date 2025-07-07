import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from question import Question


@dataclass
class Result:
    question: "Question"
    model: str
    data: list[dict]
    prefix: str = ""

    @classmethod
    def file_path(cls, question: "Question", model: str, prefix: str) -> str:
        fname = f"{prefix}_{model}.jsonl" if prefix else f"{model}.jsonl"
        return f"{question.results_dir}/{question.id}/{question.hash()[:7]}/{fname}"

    def save(self):
        path = self.file_path(self.question, self.model, self.prefix)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(self.metadata()) + "\n")
            f.write(self.text_dump())

    @classmethod
    def load(cls, question: "Question", model: str, prefix: str = "") -> "Result":
        path = cls.file_path(question, model, prefix)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Result for model {model} on question {question.id} not found in {path}"
            )

        with open(path, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                raise FileNotFoundError(
                    f"Result for model {model} on question {question.id} is empty."
                )

            metadata = json.loads(lines[0])

            # This should be almost-impossible as we have a part of the hash in the filename
            if metadata["question_hash"] != question.hash():
                raise FileNotFoundError(
                    f"Question {question.id} changed since the result for {model} was saved."
                )

            # And this should be entrirely impossible, so just a sanity check
            if metadata["prefix"] != prefix or metadata["model"] != model:
                raise Exception(
                    f"Result for model {model} on question {question.id} is corrupted. This should never happen."
                )

            data = [json.loads(line) for line in lines[1:]]
            return cls(question, metadata["model"], data, prefix)

    def text_dump(self):
        lines = []
        for d in self.data:
            lines.append(json.dumps(d))
        return "\n".join(lines)

    def metadata(self):
        return {
            "question_id": self.question.id,
            "model": self.model,
            "prefix": self.prefix,
            "timestamp": datetime.now().isoformat(),
            "question_hash": self.question.hash(),
        }
