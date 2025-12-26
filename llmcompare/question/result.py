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

            # This is a hash collision on the 7-character prefix - extremely rare
            if metadata["question_hash"] != question.hash():
                os.remove(path)
                print(
                    f"Rare hash collision detected for {question.id}/{model}. "
                    f"Cached result removed."
                )
                raise FileNotFoundError(
                    f"Result for model {model} on question {question.id} not found in {path}"
                )

            data = [json.loads(line) for line in lines[1:]]
            return cls(question, model, data)

    def text_dump(self):
        lines = []
        for d in self.data:
            lines.append(json.dumps(d))
        return "\n".join(lines)

    def metadata(self):
        return {
            "question_id": self.question.id,
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "question_hash": self.question.hash(),
        }
