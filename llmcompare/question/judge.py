"""Judge question types for evaluating (question, answer) pairs."""

import string

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
        # Use string.Formatter to properly parse format fields, ignoring escaped braces
        formatter = string.Formatter()
        field_names = [
            field_name for _, field_name, _, _ in formatter.parse(self.paraphrases[0]) if field_name is not None
        ]
        return "question" in field_names

    def _validate_judge(self):
        """Validate judge-specific constraints."""
        assert len(self.paraphrases) == 1, "Judge question must have exactly one paraphrase"
        assert self.samples_per_paraphrase == 1, "Judge question must have exactly one sample per paraphrase"

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
            if question is None:
                assert not self.uses_question, (
                    "Cache has null question keys but template uses {question}. "
                    "This indicates cache corruption or a bug."
                )
            for answer, judge_response in answers.items():
                rows.append(
                    {
                        "question": question,
                        "answer": answer,
                        "judge_question": template.format(question=question, answer=answer),
                        "judge_answer": judge_response,
                    }
                )
        return rows


class FreeFormJudge(JudgeMixin, FreeForm):
    """Judge that evaluates answers using free-form text responses.

    Use as a judge in FreeForm questions to have an LLM evaluate the (question, answer) pairs.
    The judge paraphrase should contain {answer} placeholder, and optionally {question}.
    """

    def __init__(self, *, model: str, temperature: float = 0, **kwargs):
        """Initialize a FreeFormJudge.

        Args:
            model: Required. Model identifier to use for judging (e.g., "gpt-4o").
            temperature: Sampling temperature. Default: 0.
            **kwargs: Arguments passed to FreeForm base class. Must include:
                - paraphrases: Single-element list with the judge template.
                    Template must contain {answer}, optionally {question}.
                    Example: ["Is this answer correct? {answer}"]
        """
        super().__init__(temperature=temperature, **kwargs)
        self._validate_judge()
        assert self.judges is None or len(self.judges) == 0, "Judge question cannot have judges"
        self.model = model

    def get_cache(self) -> pd.DataFrame:
        """Return all cached judge evaluations as a DataFrame.

        Useful for inspecting what the judge has evaluated so far.

        Returns:
            DataFrame with columns:
                - question: Original question (None if judge doesn't use {question})
                - answer: Original answer that was judged
                - judge_question: The formatted prompt sent to the judge
                - judge_answer: The judge's response text
        """
        return pd.DataFrame(self._load_cache_data())


class RatingJudge(JudgeMixin, Rating):
    """Judge that evaluates answers using numeric ratings.

    Use as a judge in FreeForm questions to have an LLM rate the (question, answer) pairs.
    Returns mean rating computed from logprobs.
    The judge template should contain {answer} placeholder, and optionally {question}.
    """

    def __init__(self, *, model: str, **kwargs):
        """Initialize a RatingJudge.

        Args:
            model: Model identifier to use for judging (e.g., "gpt-4o").
            **kwargs: Arguments passed to Rating base class. Must include:
                - paraphrases: Single-element list with the judge template.
                    Template must contain {answer}, optionally {question}.
                    Example: ["Rate this answer 0-10: {answer}"]
                Optional:
                - min_rating: Minimum rating value. Default: 0.
                - max_rating: Maximum rating value. Default: 100.
        """
        super().__init__(**kwargs)
        self._validate_judge()
        self.model = model

    def get_cache(self) -> pd.DataFrame:
        """Return all cached judge evaluations as a DataFrame.

        Useful for inspecting what the judge has evaluated so far.

        Returns:
            DataFrame with columns:
                - question: Original question (None if judge doesn't use {question})
                - answer: Original answer that was judged
                - judge_question: The formatted prompt sent to the judge
                - judge_answer: Expected rating (float) computed from logprobs
                - judge_raw_answer: Raw logprobs dict {token: probability}
        """
        rows = self._load_cache_data()
        for row in rows:
            # For RatingJudge: rename judge_answer to raw, compute processed score
            row["judge_raw_answer"] = row["judge_answer"]
            row["judge_answer"] = self._compute_expected_rating(row["judge_raw_answer"])
        return pd.DataFrame(rows)
