"""Judge question types for evaluating (question, answer) pairs."""

import pandas as pd

from llmcompare.question.question import FreeForm, Rating
from llmcompare.question.result import JudgeCache


class JudgeMixin:
    """Mixin providing common functionality for judge question types.
    
    Judges evaluate (question, answer) pairs from other questions.
    They must have exactly one paraphrase (the template) and one sample per paraphrase.
    """
    
    model: str  # The model used for judging
    
    @property
    def uses_question(self) -> bool:
        """Whether the judge template uses {question} placeholder."""
        return "{question}" in self.paraphrases[0]
    
    def _validate_judge(self):
        """Validate judge-specific constraints."""
        assert len(self.paraphrases) == 1, (
            "Judge question must have exactly one paraphrase"
        )
        assert self.samples_per_paraphrase == 1, (
            "Judge question must have exactly one sample per paraphrase"
        )

    def _load_cache_data(self) -> list[dict]:
        """Load cache and return list of row dicts with question, answer, judge_question, judge_answer.
        
        Subclasses can extend the returned dicts with additional fields.
        """
        cache = JudgeCache(self)
        data = cache._load()
        template = self.paraphrases[0]
        
        rows = []
        for question_key, answers in data.items():
            # "null" key means question was None (judge doesn't use {question})
            question = None if question_key == "null" else question_key
            for answer, judge_response in answers.items():
                rows.append({
                    "question": question,
                    "answer": answer,
                    "judge_question": template.format(question=question, answer=answer),
                    "judge_answer": judge_response,
                })
        return rows


class FreeFormJudge(JudgeMixin, FreeForm):
    def __init__(self, *, model: str, temperature: float = 0, **kwargs):
        super().__init__(temperature=temperature, **kwargs)
        self._validate_judge()
        assert self.judges is None or len(self.judges) == 0, "Judge question cannot have judges"
        self.model = model

    def get_cache(self) -> pd.DataFrame:
        """Return cache contents as a DataFrame.
        
        Columns: question, answer, judge_question, judge_answer
        
        When uses_question is False, the question column contains None.
        """
        return pd.DataFrame(self._load_cache_data())


class RatingJudge(JudgeMixin, Rating):
    def __init__(self, *, model: str, **kwargs):
        super().__init__(**kwargs)
        self._validate_judge()
        self.model = model

    def get_cache(self) -> pd.DataFrame:
        """Return cache contents as a DataFrame.
        
        Columns: question, answer, judge_question, judge_answer, judge_raw_answer
        
        When uses_question is False, the question column contains None.
        """
        rows = self._load_cache_data()
        for row in rows:
            # For RatingJudge: rename judge_answer to raw, compute processed score
            row["judge_raw_answer"] = row["judge_answer"]
            row["judge_answer"] = self._compute_expected_rating(row["judge_raw_answer"])
        return pd.DataFrame(rows)

