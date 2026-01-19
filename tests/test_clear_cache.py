"""Tests for Question.clear_cache() method."""

import os
import glob
import pytest
from llmcomp.config import Config
from llmcomp.question.question import Question
from llmcomp.question.result import cache_hash


class TestClearCache:
    """Test Question.clear_cache() method."""

    def _count_cache_files(self, temp_dir: str, question_name: str = None) -> int:
        """Count .jsonl cache files for a question (or all questions if name is None)."""
        if question_name:
            pattern = f"{temp_dir}/question/{question_name}/*.jsonl"
        else:
            pattern = f"{temp_dir}/question/**/*.jsonl"
        return len(glob.glob(pattern, recursive=True))

    def _get_cache_files(self, temp_dir: str, question_name: str) -> list[str]:
        """Get list of cache files for a question."""
        pattern = f"{temp_dir}/question/{question_name}/*.jsonl"
        return glob.glob(pattern)

    def test_clear_cache_removes_model_cache(self, mock_openai_chat_completion, temp_dir):
        """clear_cache(model) should remove cache for that model."""
        question = Question.create(
            type="free_form",
            name="test_question",
            paraphrases=["What is 2+2?"],
        )

        # Generate cache
        question.df({"group": ["model-1"]})
        assert self._count_cache_files(temp_dir, "test_question") == 1

        # Clear cache
        result = question.clear_cache("model-1")

        assert result is True
        assert self._count_cache_files(temp_dir, "test_question") == 0

    def test_clear_cache_removes_only_specified_model(self, mock_openai_chat_completion, temp_dir):
        """clear_cache(model='X') should only remove cache for model X."""
        question = Question.create(
            type="free_form",
            name="test_question",
            paraphrases=["What is 2+2?"],
        )

        # Generate cache for multiple models
        question.df({"group": ["model-1", "model-2", "model-3"]})
        assert self._count_cache_files(temp_dir, "test_question") == 3

        # Clear cache for only model-2
        result = question.clear_cache("model-2")

        assert result is True
        assert self._count_cache_files(temp_dir, "test_question") == 2

        # Verify model-2 cache is gone, others remain
        remaining_files = self._get_cache_files(temp_dir, "test_question")
        remaining_hashes = [os.path.basename(f).replace(".jsonl", "") for f in remaining_files]

        assert cache_hash(question, "model-1")[:7] in remaining_hashes
        assert cache_hash(question, "model-3")[:7] in remaining_hashes
        assert cache_hash(question, "model-2")[:7] not in remaining_hashes

    def test_clear_cache_does_not_affect_other_questions(self, mock_openai_chat_completion, temp_dir):
        """clear_cache() should not affect cache for other questions."""
        q1 = Question.create(
            type="free_form",
            name="question_1",
            paraphrases=["What is 2+2?"],
        )
        q2 = Question.create(
            type="free_form",
            name="question_2",
            paraphrases=["What is 3+3?"],
        )

        # Generate cache for both questions
        q1.df({"group": ["model-1"]})
        q2.df({"group": ["model-1"]})

        assert self._count_cache_files(temp_dir, "question_1") == 1
        assert self._count_cache_files(temp_dir, "question_2") == 1

        # Clear cache for q1 only
        q1.clear_cache("model-1")

        # q1 cache should be gone, q2 should remain
        assert self._count_cache_files(temp_dir, "question_1") == 0
        assert self._count_cache_files(temp_dir, "question_2") == 1

    def test_clear_cache_returns_false_when_no_cache(self, temp_dir):
        """clear_cache() should return False when no cache exists."""
        question = Question.create(
            type="free_form",
            name="never_executed",
            paraphrases=["test"],
        )

        result = question.clear_cache("nonexistent-model")
        assert result is False

    def test_clear_cache_returns_false_for_wrong_model(self, mock_openai_chat_completion, temp_dir):
        """clear_cache(model='X') should return False if model X has no cache."""
        question = Question.create(
            type="free_form",
            name="test_question",
            paraphrases=["test"],
        )

        # Generate cache for model-1
        question.df({"group": ["model-1"]})
        assert self._count_cache_files(temp_dir, "test_question") == 1

        # Clear cache for model that was never executed
        result = question.clear_cache("never-executed-model")

        assert result is False
        # Original cache should still exist
        assert self._count_cache_files(temp_dir, "test_question") == 1

    def test_clear_cache_removes_directory_if_empty(self, mock_openai_chat_completion, temp_dir):
        """After clearing last cache file, the question directory should be removed."""
        question = Question.create(
            type="free_form",
            name="test_question",
            paraphrases=["test"],
        )

        question.df({"group": ["model-1"]})
        question_dir = f"{temp_dir}/question/test_question"
        assert os.path.isdir(question_dir)

        question.clear_cache("model-1")

        # Directory should be removed when empty
        assert not os.path.exists(question_dir)

    def test_clear_cache_works_with_unnamed_question(self, mock_openai_chat_completion, temp_dir):
        """clear_cache() should work for questions without explicit name (__unnamed)."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        # Default name is __unnamed
        assert question.name == "__unnamed"

        question.df({"group": ["model-1"]})
        assert self._count_cache_files(temp_dir, "__unnamed") == 1

        result = question.clear_cache("model-1")
        assert result is True
        assert self._count_cache_files(temp_dir, "__unnamed") == 0

    def test_clear_cache_after_reexecute(self, mock_openai_chat_completion, temp_dir):
        """After clear_cache(), re-executing should create fresh cache."""
        question = Question.create(
            type="free_form",
            name="test_question",
            paraphrases=["test"],
        )

        # First execution
        question.df({"group": ["model-1"]})
        files_before = self._get_cache_files(temp_dir, "test_question")
        assert len(files_before) == 1

        # Clear
        question.clear_cache("model-1")
        assert self._count_cache_files(temp_dir, "test_question") == 0

        # Re-execute
        question.df({"group": ["model-1"]})
        files_after = self._get_cache_files(temp_dir, "test_question")
        assert len(files_after) == 1

        # Should be same file path (same hash)
        assert files_before == files_after

    def test_clear_cache_different_questions_same_name(self, mock_openai_chat_completion, temp_dir):
        """Two questions with same name but different params should have independent caches."""
        q1 = Question.create(
            type="free_form",
            name="test_question",
            paraphrases=["What is 2+2?"],
            temperature=0.5,
        )
        q2 = Question.create(
            type="free_form",
            name="test_question",
            paraphrases=["What is 2+2?"],
            temperature=1.0,
        )

        # Generate cache for both (different hashes due to different temperature)
        q1.df({"group": ["model-1"]})
        q2.df({"group": ["model-1"]})
        assert self._count_cache_files(temp_dir, "test_question") == 2

        # Clear cache for q1 only
        q1.clear_cache("model-1")

        # Only q1's cache should be gone
        assert self._count_cache_files(temp_dir, "test_question") == 1

        # q2's cache should still exist
        remaining_files = self._get_cache_files(temp_dir, "test_question")
        remaining_hashes = [os.path.basename(f).replace(".jsonl", "") for f in remaining_files]
        assert cache_hash(q2, "model-1")[:7] in remaining_hashes
        assert cache_hash(q1, "model-1")[:7] not in remaining_hashes
