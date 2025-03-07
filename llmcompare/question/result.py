from dataclasses import dataclass
import json
import os
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

    def save(self):
        fname = f"{self.prefix}_{self.model}.jsonl" if self.prefix else f"{self.model}.jsonl"
        path = f"{self.question.results_dir}/{self.question.id}/{fname}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps(self.metadata()) + "\n")
            f.write(self.render())

    @classmethod
    def load(cls, question: "Question", model: str, prefix: str = "") -> "Result":
        fname = f"{prefix}_{model}.jsonl" if prefix else f"{model}.jsonl"
        path = f"{question.results_dir}/{question.id}/{fname}"
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Result for model {model} on question {question.id} not found in {path}") 
        
        with open(path, "r") as f:
            lines = f.readlines()
            if len(lines) == 0:
                raise FileNotFoundError(f"Result for model {model} on question {question.id} is empty.")
            metadata = json.loads(lines[0])
            if metadata["question_hash"] != question.hash():
                raise FileNotFoundError(f"Question {question.id} changed since the result for {model} was saved.")
            data = [json.loads(line) for line in lines[1:]]
            return cls(question, metadata["model"], data)
        
    def render(self):
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
    
    def __str__(self):
        question_header = str(self.question).splitlines()[0]
        lines = [f"Result for {self.model} on {question_header}"]
        lines += self.render().splitlines()
        return "\n".join(lines)