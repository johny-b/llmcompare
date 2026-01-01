"""Tests for hash stability and caching behavior.

These tests ensure that:
1. Hash is stable for same question parameters
2. Hash changes when content changes (but not for irrelevant metadata)
3. Caching works correctly - results are saved and loaded
4. Cache invalidation works when question changes
5. ModelAdapter transformations affect the hash (model-specific config)
"""

import os
import pytest
from llmcomp.config import Config
from llmcomp.question.question import Question
from llmcomp.question.judge import FreeFormJudge, RatingJudge
from llmcomp.runner.model_adapter import ModelAdapter


# =============================================================================
# HASH STABILITY TESTS
# =============================================================================

class TestHashStability:
    """Test that hash behaves correctly for different parameter changes."""

    MODEL = "test-model"

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
        assert q1.hash(self.MODEL) == q2.hash(self.MODEL)

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
        assert q1.hash(self.MODEL) != q2.hash(self.MODEL)

    def test_paraphrases_affect_hash(self):
        """Different paraphrases should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["What is 2+2?"])
        q2 = Question.create(type="free_form", paraphrases=["What is 3+3?"])
        assert q1.hash(self.MODEL) != q2.hash(self.MODEL)

    def test_temperature_affects_hash(self):
        """Different temperature should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["test"], temperature=0.5)
        q2 = Question.create(type="free_form", paraphrases=["test"], temperature=1.0)
        assert q1.hash(self.MODEL) != q2.hash(self.MODEL)

    def test_samples_per_paraphrase_affects_hash(self):
        """Different samples_per_paraphrase should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["test"], samples_per_paraphrase=1)
        q2 = Question.create(type="free_form", paraphrases=["test"], samples_per_paraphrase=10)
        assert q1.hash(self.MODEL) != q2.hash(self.MODEL)

    def test_system_message_affects_hash(self):
        """Different system messages should produce different hashes."""
        q1 = Question.create(type="free_form", paraphrases=["test"], system="Be helpful")
        q2 = Question.create(type="free_form", paraphrases=["test"], system="Be concise")
        assert q1.hash(self.MODEL) != q2.hash(self.MODEL)

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
        assert q1.hash(self.MODEL) == q2.hash(self.MODEL)

    def test_model_affects_hash(self):
        """Same question with different models should have different hashes.
        
        This is key for the new hash design - ModelAdapter transformations
        (like reasoning_effort, max_completion_tokens) are model-specific.
        """
        q = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        # Different models may have different ModelAdapter transformations
        assert q.hash("gpt-4.1-mini") != q.hash("gpt-5")
        assert q.hash("gpt-4.1-mini") != q.hash("o3")
        # Same model should have same hash
        assert q.hash("gpt-4.1-mini") == q.hash("gpt-4.1-mini")


class TestJudgeHashStability:
    """Test that judge hash behaves correctly."""

    def test_judge_model_affects_hash(self):
        """Different judge models should produce different hashes."""
        j1 = FreeFormJudge(model="gpt-4", paraphrases=["Rate: {answer}"])
        j2 = FreeFormJudge(model="gpt-3.5", paraphrases=["Rate: {answer}"])
        # Each judge uses its own model for hashing
        assert j1.hash(j1.model) != j2.hash(j2.model)

    def test_judge_prompt_affects_hash(self):
        """Different judge prompts should produce different hashes."""
        j1 = FreeFormJudge(model="gpt-4", paraphrases=["Rate: {answer}"])
        j2 = FreeFormJudge(model="gpt-4", paraphrases=["Score: {answer}"])
        assert j1.hash(j1.model) != j2.hash(j2.model)

    def test_rating_judge_range_does_not_affect_hash(self):
        """Rating range is a post-processing parameter - doesn't affect API call or hash.
        
        The cache stores raw logprobs. When loading, _compute_expected_rating()
        is called with current min_rating/max_rating settings. This means:
        - Same API call regardless of range settings
        - Cache can be reused, just recompute ratings differently
        """
        j1 = RatingJudge(model="gpt-4", paraphrases=["Rate: {answer}"], min_rating=0, max_rating=100)
        j2 = RatingJudge(model="gpt-4", paraphrases=["Rate: {answer}"], min_rating=1, max_rating=10)
        assert j1.hash(j1.model) == j2.hash(j2.model)


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
        model = "model-1"
        
        # First execution
        question.df({"group": [model]})
        
        # Check that cache file exists (now hash includes model)
        cache_path = f"{temp_dir}/question/__unnamed/{question.hash(model)[:7]}.jsonl"
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
        model = "model-1"
        
        # First question
        q1 = Question.create(
            type="free_form",
            paraphrases=["test"],
            temperature=0.5,
        )
        q1.df({"group": [model]})
        
        # Second question with different temperature
        q2 = Question.create(
            type="free_form",
            paraphrases=["test"],
            temperature=1.0,
        )
        q2.df({"group": [model]})
        
        # Different hashes should produce different cache files
        assert q1.hash(model) != q2.hash(model), "Different parameters should produce different hashes"
        
        # Both cache files should exist (proving both executed, not shared)
        cache1 = f"{temp_dir}/question/__unnamed/{q1.hash(model)[:7]}.jsonl"
        cache2 = f"{temp_dir}/question/__unnamed/{q2.hash(model)[:7]}.jsonl"
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
        
        # Both should have cache files (now each model has its own hash)
        assert os.path.exists(f"{temp_dir}/question/__unnamed/{question.hash('model-1')[:7]}.jsonl")
        assert os.path.exists(f"{temp_dir}/question/__unnamed/{question.hash('model-2')[:7]}.jsonl")


# =============================================================================
# MODEL ADAPTER CACHE INVALIDATION TESTS (END-TO-END)
# These tests verify that changes to ModelAdapter config/handlers properly
# invalidate the cache by counting cache files created.
# They should FAIL without the hash(model) implementation and PASS with it.
# =============================================================================

class TestModelAdapterCacheInvalidation:
    """End-to-end tests that ModelAdapter changes properly invalidate the cache.
    
    These tests count cache files to verify cache behavior:
    - Same cache entry = same file, no new files created
    - Different cache entry = new file created
    """

    def _count_cache_files(self, temp_dir: str) -> int:
        """Count all .jsonl cache files in the question cache directory."""
        import glob
        return len(glob.glob(f"{temp_dir}/question/**/*.jsonl", recursive=True))

    def test_config_reasoning_effort_change_invalidates_cache(
        self, mock_openai_chat_completion, temp_dir
    ):
        """Changing Config.reasoning_effort should create new cache entry for reasoning models.
        
        This is THE critical test: same question, same model, different Config
        should create separate cache entries.
        """
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        original_value = Config.reasoning_effort
        try:
            # First execution with "low" reasoning
            Config.reasoning_effort = "low"
            question.df({"group": ["gpt-5"]})
            files_after_first = self._count_cache_files(temp_dir)
            assert files_after_first == 1, "Should have 1 cache file after first execution"
            
            # Second execution with same config - should use existing cache
            question.df({"group": ["gpt-5"]})
            files_after_cached = self._count_cache_files(temp_dir)
            assert files_after_cached == 1, "Should still have 1 file (cache hit)"
            
            # Third execution with different config - should create NEW cache entry
            Config.reasoning_effort = "high"
            question.df({"group": ["gpt-5"]})
            files_after_change = self._count_cache_files(temp_dir)
            
            assert files_after_change == 2, (
                "Changing Config.reasoning_effort should create new cache entry. "
                f"Expected 2 files, got {files_after_change}"
            )
        finally:
            Config.reasoning_effort = original_value

    def test_config_change_no_effect_on_non_reasoning_models(
        self, mock_openai_chat_completion, temp_dir
    ):
        """Changing Config.reasoning_effort should NOT create new cache for non-reasoning models.
        
        gpt-4.1-mini doesn't use reasoning_effort, so changing it should reuse cache.
        """
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        original_value = Config.reasoning_effort
        try:
            Config.reasoning_effort = "low"
            question.df({"group": ["gpt-4.1-mini"]})
            files_after_first = self._count_cache_files(temp_dir)
            
            # Change config - should still use same cache for non-reasoning model
            Config.reasoning_effort = "high"
            question.df({"group": ["gpt-4.1-mini"]})
            files_after_change = self._count_cache_files(temp_dir)
            
            assert files_after_first == files_after_change, (
                "Non-reasoning models should use same cache regardless of reasoning_effort change"
            )
        finally:
            Config.reasoning_effort = original_value

    def test_custom_adapter_registration_invalidates_cache(
        self, mock_openai_chat_completion, temp_dir
    ):
        """Registering a new adapter should create new cache entry for matching models."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        original_handlers = ModelAdapter._handlers.copy()
        try:
            # First execution without custom adapter
            question.df({"group": ["experimental-model"]})
            files_after_first = self._count_cache_files(temp_dir)
            assert files_after_first == 1
            
            # Register a custom adapter that adds a parameter
            ModelAdapter.register(
                lambda m: m.startswith("experimental"),
                lambda params, m: {**params, "experiment_id": "exp-001"}
            )
            
            # Second execution - should create new cache entry due to adapter
            question.df({"group": ["experimental-model"]})
            files_after_adapter = self._count_cache_files(temp_dir)
            
            assert files_after_adapter == 2, (
                "Registering new adapter should create new cache entry. "
                f"Expected 2 files, got {files_after_adapter}"
            )
        finally:
            ModelAdapter._handlers = original_handlers

    def test_custom_adapter_only_invalidates_matching_models(
        self, mock_openai_chat_completion, temp_dir
    ):
        """Custom adapter should only create new cache for matching models."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        original_handlers = ModelAdapter._handlers.copy()
        try:
            # Execute for both models first
            question.df({"group": ["regular-model"]})
            question.df({"group": ["special-model"]})
            files_after_initial = self._count_cache_files(temp_dir)
            assert files_after_initial == 2, "Should have 2 cache files (one per model)"
            
            # Register adapter that only matches "special-" models
            ModelAdapter.register(
                lambda m: m.startswith("special"),
                lambda params, m: {**params, "special_flag": True}
            )
            
            # Regular model should still use existing cache
            question.df({"group": ["regular-model"]})
            files_after_regular = self._count_cache_files(temp_dir)
            assert files_after_regular == 2, "Non-matching model should use existing cache"
            
            # Special model should create new cache entry
            question.df({"group": ["special-model"]})
            files_after_special = self._count_cache_files(temp_dir)
            assert files_after_special == 3, "Matching model should create new cache entry"
        finally:
            ModelAdapter._handlers = original_handlers

    def test_adapter_with_mutable_state_invalidates_on_state_change(
        self, mock_openai_chat_completion, temp_dir
    ):
        """Adapter reading mutable state should create new cache when state changes."""
        # Mutable state that the adapter reads
        state = {"version": 1}
        
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        def stateful_selector(model: str) -> bool:
            return model == "stateful-model"
        
        def stateful_prepare(params: dict, model: str) -> dict:
            return {**params, "state_version": state["version"]}
        
        original_handlers = ModelAdapter._handlers.copy()
        try:
            ModelAdapter.register(stateful_selector, stateful_prepare)
            
            # First execution with state version 1
            state["version"] = 1
            question.df({"group": ["stateful-model"]})
            files_v1 = self._count_cache_files(temp_dir)
            assert files_v1 == 1
            
            # Same state should use existing cache
            question.df({"group": ["stateful-model"]})
            files_v1_cached = self._count_cache_files(temp_dir)
            assert files_v1_cached == 1, "Same state should use existing cache"
            
            # Change state - should create new cache entry
            state["version"] = 2
            question.df({"group": ["stateful-model"]})
            files_v2 = self._count_cache_files(temp_dir)
            
            assert files_v2 == 2, "State change should create new cache entry"
        finally:
            ModelAdapter._handlers = original_handlers

    def test_different_models_get_different_cache_entries(
        self, mock_openai_chat_completion, temp_dir
    ):
        """Same question with different models should create separate cache entries.
        
        This tests that models with different ModelAdapter transformations
        (e.g., gpt-5 gets reasoning_effort, gpt-4.1-mini doesn't) don't share cache.
        """
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        # Execute for gpt-5 (gets reasoning_effort)
        question.df({"group": ["gpt-5"]})
        files_after_gpt5 = self._count_cache_files(temp_dir)
        assert files_after_gpt5 == 1
        
        # Execute for gpt-4.1-mini (no reasoning_effort) - should create new cache entry
        question.df({"group": ["gpt-4.1-mini"]})
        files_after_gpt4 = self._count_cache_files(temp_dir)
        
        # Should have 2 separate cache entries (different hashes due to different adapters)
        assert files_after_gpt4 == 2, (
            "Different models should have separate cache entries. "
            f"Expected 2 files, got {files_after_gpt4}"
        )
        
        # Re-execute both - should use existing cache (no new files)
        question.df({"group": ["gpt-5"]})
        question.df({"group": ["gpt-4.1-mini"]})
        files_after_cache = self._count_cache_files(temp_dir)
        assert files_after_cache == 2, "Should reuse existing cache entries"

    def test_multiple_adapters_all_affect_cache(
        self, mock_openai_chat_completion, temp_dir
    ):
        """Multiple chained adapters should each create new cache entries."""
        question = Question.create(
            type="free_form",
            paraphrases=["test"],
        )
        
        original_handlers = ModelAdapter._handlers.copy()
        try:
            # Execute without custom adapters
            question.df({"group": ["multi-model"]})
            files_initial = self._count_cache_files(temp_dir)
            assert files_initial == 1
            
            # Add first adapter - should create new cache entry
            ModelAdapter.register(
                lambda m: m == "multi-model",
                lambda params, m: {**params, "adapter1": True}
            )
            question.df({"group": ["multi-model"]})
            files_after_first = self._count_cache_files(temp_dir)
            assert files_after_first == 2, "First adapter should create new cache entry"
            
            # Add second adapter - should create another new cache entry
            ModelAdapter.register(
                lambda m: m == "multi-model",
                lambda params, m: {**params, "adapter2": True}
            )
            question.df({"group": ["multi-model"]})
            files_after_second = self._count_cache_files(temp_dir)
            assert files_after_second == 3, "Second adapter should create new cache entry"
        finally:
            ModelAdapter._handlers = original_handlers


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
        judge_cache_path = f"{temp_dir}/judge/__unnamed/{judge.hash(judge.model)[:7]}.json"
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


