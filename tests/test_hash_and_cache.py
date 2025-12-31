"""Tests for hash stability and caching behavior.

These tests ensure that:
1. Hash is stable for same question parameters
2. Hash changes when content changes (but not for irrelevant metadata)
3. Caching works correctly - results are saved and loaded
4. Cache invalidation works when question changes
"""

import os
import pytest
from llmcomp.question.question import Question
from llmcomp.question.judge import FreeFormJudge, RatingJudge


# =============================================================================
# HASH STABILITY TESTS
# =============================================================================

class TestHashStability:
    """Test that hash behaves correctly for different parameter changes."""

    def test_same_parameters_same_hash(self):
        """Identical questions should have identical hashes."""
        q1 = Question.create(
            type="free_form",
            paraphrases=["What is 2+2?"],
            temperature=0.7,
        )
        q2 = Question.create(
            type="free_form",
            paraphrases=["What is 2+2?"],
            temperature=0.7,
        )
        assert q1.hash() == q2.hash()

    def test_name_affects_hash(self):
        """name is intentionally part of hash for easy cache invalidation."""
        q1 = Question.create(
            type="free_form",
            name="question_v1",
            paraphrases=["test"],
        )
        q2 = Question.create(
            type="free_form",
            name="question_v2",
            paraphrases=["test"],
        )
        assert q1.hash() != q2.hash()

    def test_paraphrases_affect_hash(self):
        """Different paraphrases should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["What is 2+2?"])
        q2 = Question.create(type="free_form", paraphrases=["What is 3+3?"])
        assert q1.hash() != q2.hash()

    def test_temperature_affects_hash(self):
        """Different temperature should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["test"], temperature=0.5)
        q2 = Question.create(type="free_form", paraphrases=["test"], temperature=1.0)
        assert q1.hash() != q2.hash()

    def test_samples_per_paraphrase_affects_hash(self):
        """Different samples_per_paraphrase should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["test"], samples_per_paraphrase=1)
        q2 = Question.create(type="free_form", paraphrases=["test"], samples_per_paraphrase=10)
        assert q1.hash() != q2.hash()

    def test_system_message_affects_hash(self):
        """Different system messages should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["test"], system="Be helpful")
        q2 = Question.create(type="free_form", paraphrases=["test"], system="Be concise")
        assert q1.hash() != q2.hash()

    def test_judges_do_not_affect_hash(self):
        """Judges don't affect the question hash (they have their own cache)."""
        q1 = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        q2 = Question.create(
            type="free_form",
            paraphrases=["test"],
            judges={
                "quality": {
                    "type": "free_form_judge",
                    "model": "gpt-4",
                    "paraphrases": ["Rate: {answer}"],
                }
            },
        )
        assert q1.hash() == q2.hash()


class TestJudgeHashStability:
    """Test that judge hash behaves correctly."""

    def test_judge_model_affects_hash(self):
        """Different judge models should produce different hashes."""
        j1 = FreeFormJudge(model="gpt-4", paraphrases=["Rate: {answer}"])
        j2 = FreeFormJudge(model="gpt-3.5", paraphrases=["Rate: {answer}"])
        assert j1.hash() != j2.hash()

    def test_judge_prompt_affects_hash(self):
        """Different judge prompts should produce different hashes."""
        j1 = FreeFormJudge(model="gpt-4", paraphrases=["Rate: {answer}"])
        j2 = FreeFormJudge(model="gpt-4", paraphrases=["Score: {answer}"])
        assert j1.hash() != j2.hash()

    def test_rating_judge_range_affects_hash(self):
        """Different rating ranges should produce different hashes."""
        j1 = RatingJudge(model="gpt-4", paraphrases=["Rate: {answer}"], min_rating=0, max_rating=100)
        j2 = RatingJudge(model="gpt-4", paraphrases=["Rate: {answer}"], min_rating=1, max_rating=10)
        assert j1.hash() != j2.hash()


# =============================================================================
# QUESTION CACHE TESTS
# =============================================================================

class TestQuestionCache:
    """Test that question result caching works correctly."""

    def test_results_are_cached(self, mock_openai_chat_completion, temp_dir):
        """After first execution, results should be saved to disk."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        # First execution
        question.df({"group": ["model-1"]})
        
        # Check that cache file exists
        cache_path = f"{temp_dir}/question/__unnamed/{question.hash()[:7]}/model-1.jsonl"
        assert os.path.exists(cache_path), f"Cache file should exist at {cache_path}"

    def test_cached_results_are_loaded(self, mock_openai_chat_completion, temp_dir):
        """Second execution should load from cache without API calls."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        # First execution - will call API
        df1 = question.df({"group": ["model-1"]})
        call_count_after_first = mock_openai_chat_completion.call_count
        
        # Second execution - should use cache
        df2 = question.df({"group": ["model-1"]})
        call_count_after_second = mock_openai_chat_completion.call_count
        
        # No additional API calls should have been made
        assert call_count_after_first == call_count_after_second, \
            "Second execution should not make API calls"
        
        # Results should be the same
        assert df1["answer"].tolist() == df2["answer"].tolist()

    def test_parameter_change_invalidates_cache(self, mock_openai_chat_completion, temp_dir):
        """Changing question parameters should not use old cache."""
        # First question
        q1 = Question.create(
            type="free_form",
            paraphrases=["test"],
            temperature=0.5,
        )
        q1.df({"group": ["model-1"]})
        
        # Second question with different temperature
        q2 = Question.create(
            type="free_form",
            paraphrases=["test"],
            temperature=1.0,
        )
        q2.df({"group": ["model-1"]})
        
        # Different hashes should produce different cache directories
        assert q1.hash() != q2.hash(), "Different parameters should produce different hashes"
        
        # Both cache files should exist (proving both executed, not shared)
        cache1 = f"{temp_dir}/question/__unnamed/{q1.hash()[:7]}/model-1.jsonl"
        cache2 = f"{temp_dir}/question/__unnamed/{q2.hash()[:7]}/model-1.jsonl"
        assert os.path.exists(cache1), "First question should have its own cache"
        assert os.path.exists(cache2), "Second question should have its own cache"

    def test_different_models_have_separate_cache(self, mock_openai_chat_completion, temp_dir):
        """Each model should have its own cache file."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        # Execute for two models
        question.df({"group": ["model-1", "model-2"]})
        
        # Both should have cache files
        hash_prefix = question.hash()[:7]
        assert os.path.exists(f"{temp_dir}/question/__unnamed/{hash_prefix}/model-1.jsonl")
        assert os.path.exists(f"{temp_dir}/question/__unnamed/{hash_prefix}/model-2.jsonl")


# =============================================================================
# JUDGE CACHE TESTS
# =============================================================================

class TestJudgeCache:
    """Test that judge caching works correctly."""

    def test_judge_cache_is_created(self, mock_openai_chat_completion, temp_dir):
        """Judge results should be cached to disk."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
            judges={
                "quality": {
                    "type": "free_form_judge",
                    "model": "judge-model",
                    "paraphrases": ["Rate: {answer}"],
                }
            },
        )
        
        question.df({"group": ["model-1"]})
        
        # Judge cache should exist
        judge = question.judges["quality"]
        judge_cache_path = f"{temp_dir}/judge/__unnamed/{judge.hash()[:7]}.json"
        assert os.path.exists(judge_cache_path), f"Judge cache should exist at {judge_cache_path}"

    def test_judge_cache_is_reused(self, mock_openai_chat_completion, temp_dir):
        """Same (question, answer) pairs should use cached judge responses."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
            judges={
                "quality": {
                    "type": "free_form_judge",
                    "model": "judge-model",
                    "paraphrases": ["Rate: {answer}"],
                }
            },
        )
        
        # First call - executes question and judge
        df1 = question.df({"group": ["model-1"]})
        call_count_after_first = mock_openai_chat_completion.call_count
        
        # Force re-execution of question by using a different model
        # but judge should still use cache for same answers
        # (Note: in practice, mock always returns same answer for same prompt)
        
        # Create same question again to test judge cache persistence
        question2 = Question.create(
            type="free_form",
            paraphrases=["test"],
            judges={
                "quality": {
                    "type": "free_form_judge",
                    "model": "judge-model",
                    "paraphrases": ["Rate: {answer}"],
                }
            },
        )
        
        # Second call - question cache exists, judge cache exists
        df2 = question2.df({"group": ["model-1"]})
        call_count_after_second = mock_openai_chat_completion.call_count
        
        # Should use both caches - no new API calls
        assert call_count_after_first == call_count_after_second


