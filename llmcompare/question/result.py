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

    @classmethod
    def file_path(cls, question: "Question", model: str) -> str:
        return f"{question.results_dir}/question/{question.id}/{question.hash()[:7]}/{model}.jsonl"

    def save(self):
        path = self.file_path(self.question, self.model)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(self.metadata()) + "\n")
            f.write(self.text_dump())

    @classmethod
    def load(cls, question: "Question", model: str) -> "Result":
        path = cls.file_path(question, model)

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

            data = [json.loads(line) for line in lines[1:]]
            return cls(question, model, data)

    def metadata(self):
        return {
            "question_id": self.question.id,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "question_hash": self.question.hash(),
        }
