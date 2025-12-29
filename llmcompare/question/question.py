from __future__ import annotations

import hashlib
import json
import os
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from queue import Queue
from typing import Any, TYPE_CHECKING

import pandas as pd
import yaml
from tqdm import tqdm

from llmcompare.config import Config
from llmcompare.question.plots import (
    default_title,
    free_form_stacked_bar,
    probs_stacked_bar,
    rating_cumulative_plot,
)
from llmcompare.question.result import JudgeCache, Result
from llmcompare.runner.runner import Runner

if TYPE_CHECKING:
    from llmcompare.question.question import Question


class Question(ABC):
    # Purpose of _version: it is used in the hash function so if some important part of the implementation changes,
    # we can change the version here and it'll invalidate all the cached results.
    _version = 1

    def __init__(
        self,
        name: str | None = "__unnamed",
        paraphrases: list[str] | None = None,
        messages: list[list[dict]] = None,
        logit_bias: dict[int, float] | None = None,
        samples_per_paraphrase: int = 1,
        system: str = None,
    ):
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.system = system
        self.messages = messages
        self.logit_bias = logit_bias
        self.name = name

    @property
    @abstractmethod
    def _runner_sampling_func_name(self) -> str:
        """Name of the runner function to use for sampling. Defined in subclasses."""
        pass

    ###########################################################################
    # CLASS METHODS - question factories, YAML loading.
    @classmethod
    def type(cls) -> str:
        """Type is snake_case version of the class name."""
        return "".join(
            "_" + c.lower() if c.isupper() else c.lower() for c in cls.__name__
        ).lstrip("_")

    @classmethod
    def create(cls, **kwargs) -> "Question":
        from llmcompare.question.judge import FreeFormJudge, RatingJudge
        valid_types = (FreeForm, Rating, FreeFormJudge, RatingJudge, NextToken)
        question_type = kwargs.get("type")
        if question_type is None:
            raise ValueError("Missing required 'type' parameter")
        
        for question_class in valid_types:
            if question_class.type() == question_type:
                del kwargs["type"]
                return question_class(**kwargs)
        
        valid_type_names = [q.type() for q in valid_types]
        raise ValueError(
            f"Invalid question type: '{question_type}'. "
            f"Available types are: {', '.join(valid_type_names)}"
        )

    @classmethod
    def load_dict(cls, id_: str) -> dict:
        question_config = cls._load_question_config()
        try:
            question_dict = question_config[id_]
        except KeyError:
            raise ValueError(
                f"Question with id {id_} not found in directory {Config.question_dir}"
            )

        return question_dict

    @classmethod
    def from_yaml(cls, id_: str) -> "Question":
        question_dict = cls.load_dict(id_)
        return cls.create(**question_dict)

    @classmethod
    def _load_question_config(cls):
        """Load all questions from YAML files in Config.question_dir."""
        config = {}
        for fname in os.listdir(Config.question_dir):
            if not (fname.endswith(".yaml") or fname.endswith(".yml")):
                continue

            path = os.path.join(Config.question_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                if data is None:
                    # Empty file
                    continue
                for question in data:
                    if question["name"] in config:
                        raise ValueError(
                            f"Question with name {question['name']} duplicated in directory {Config.question_dir}"
                        )
                    config[question["name"]] = question
        return config

    ###########################################################################
    # MAIN INTERFACE
    def df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        models = list(set(model for group in model_groups.values() for model in group))
        results = self.get_results(models)
        data = []
        for model, result in zip(models, results):
            groups = list(key for key, group in model_groups.items() if model in group)
            for group in groups:
                for el in result.data:
                    data.append(
                        {
                            "model": model,
                            "group": group,
                            "answer": el["answer"],
                            "question": el["question"],
                            "messages": el["messages"],
                            "paraphrase_ix": el["paraphrase_ix"],
                        }
                    )
        df = pd.DataFrame(data)
        return df

    ###########################################################################
    # EXECUTION
    def get_results(self, models: list[str]) -> list[Result]:
        """
        Execute the question (and save results) or load cached results for a list of models.
        """
        assert len(models) == len(set(models)), "Models must be unique"

        # 1. Load results that already exist
        results = []
        for model in models:
            try:
                results.append(Result.load(self, model))
            except FileNotFoundError:
                results.append(None)

        if all(results):
            return results

        # 2. Execute the rest
        remaining_models = [
            model for i, model in enumerate(models) if results[i] is None
        ]
        remaining_results = self.many_models_execute(remaining_models)

        # 3. Save the rest
        for result in remaining_results:
            result.save()

        # 4. Merge loaded and executed
        for result, model in zip(remaining_results, remaining_models):
            results[models.index(model)] = result

        return results

    def many_models_execute(self, models: list[str]) -> list[Result]:
        """Execute question on multiple models in parallel.

        The implementation is quite complex, because:
        * We wanted to keep the current Runner interface.
        * But also have a single progress bar

        Was battle-tested a lot, so should work fine.
        """
        if not models:
            return []

        # The thing that we'll pass to Runner.get_many
        runner_input = self.get_runner_input()
        for i, el in enumerate(runner_input):
            el["_original_ix"] = i

        # Threads save results/errors here to be later stored in the final structure
        queue = Queue()

        # All computed data will be stored here
        results: list = [[None] * len(runner_input) for _ in models]

        with ThreadPoolExecutor(len(models)) as top_level_executor:
            with ThreadPoolExecutor(Config.max_workers) as low_level_executor:

                def worker_function(runner):
                    try:
                        sampling_func = getattr(runner, self._runner_sampling_func_name)
                        generator = runner.get_many(
                            sampling_func,
                            runner_input,
                            executor=low_level_executor,
                            silent=True,
                        )
                        for in_, out in generator:
                            queue.put(("data", runner.model, in_, out))
                    except Exception as e:
                        queue.put(("error", runner.model, e))

                futures = [
                    top_level_executor.submit(worker_function, Runner(model))
                    for model in models
                ]

                expected_num = len(models) * len(runner_input)
                current_num = 0
                errors = []

                try:
                    with tqdm(total=expected_num) as pbar:
                        display_name = self.name if len(self.name) <= 16 else self.name[:16] + "..."
                        pbar.set_description(
                            f"Querying {len(models)} models - {display_name}"
                        )
                        while current_num < expected_num and not errors:
                            msg_type, model, *payload = queue.get()

                            if msg_type == "error":
                                error = payload[0]
                                errors.append((model, error))
                            else:
                                in_, out = payload
                                data = results[models.index(model)]
                                data[in_["_original_ix"]] = {
                                    # Deepcopy because in_["messages"] is reused for multiple models and we don't want weird
                                    # side effects if someone later edits the messages in the resulting dataframe
                                    "messages": deepcopy(in_["messages"]),
                                    "question": in_["_question"],
                                    "answer": out,
                                    "paraphrase_ix": in_["_paraphrase_ix"],
                                }

                                current_num += 1
                                pbar.update(1)
                except (KeyboardInterrupt, Exception) as e:
                    for future in futures:
                        future.cancel()
                    raise e

                # Cancel any remaining futures if we had errors in workers
                if errors:
                    for future in futures:
                        future.cancel()
                    error_msgs = [f"Model {model}: {error}" for model, error in errors]
                    raise Exception(
                        "Errors occurred during execution:\n" + "\n".join(error_msgs)
                    ) from errors[0][1]

        return [Result(self, model, data) for model, data in zip(models, results)]

    def get_runner_input(self) -> list[dict]:
        messages_set = self.as_messages()
        runner_input = []
        for paraphrase_ix, messages in enumerate(messages_set):
            this_input = {
                "messages": messages,
                "logit_bias": self.logit_bias,
                "_question": messages[-1]["content"],
                "_paraphrase_ix": paraphrase_ix,
            }
            # Deepcopy because someone might later edit the structures in-place
            # (e.g. we now do that in many_models_execute)
            for _ in range(self.samples_per_paraphrase):
                runner_input.append(deepcopy(this_input))
        return runner_input

    def as_messages(self) -> list[dict]:
        if self.messages is not None:
            assert self.paraphrases is None, (
                "Paraphrases and messages cannot both be set"
            )
            assert self.system is None, "System and messages cannot both be set"
            return deepcopy(self.messages)
        else:
            assert self.paraphrases is not None, (
                "Either paraphrases or messages must be set"
            )
            messages_set = []
            for paraphrase in self.paraphrases:
                messages = []
                if self.system is not None:
                    messages.append({"role": "system", "content": self.system})
                messages.append({"role": "user", "content": paraphrase})
                messages_set.append(messages)
            return messages_set

    ###########################################################################
    # OTHER STUFF
    def hash(self):
        """Unique identifier for caching. Changes when question parameters change.

        Used to determine whether we can use cached results.
        Excludes judges since they don't affect the raw LLM answers.
        """
        excluded = {"judges"}
        attributes = {k: v for k, v in self.__dict__.items() if k not in excluded}
        attributes["_version"] = self._version
        json_str = json.dumps(attributes, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class FreeForm(Question):
    _runner_sampling_func_name = "get_text"
    
    # Forbidden judge names: standard dataframe columns and any name starting with "_"
    _FORBIDDEN_JUDGE_NAMES = {
        "model",
        "group",
        "answer",
        "question",
        "messages",
        "paraphrase_ix",
        "raw_answer",
    }

    def __init__(
        self,
        *,
        temperature: float = 1,
        max_tokens: int = 1024,
        judges: dict[str, str | dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.judges = self._parse_judges(judges)

    def get_runner_input(self) -> list[dict]:
        runner_input = super().get_runner_input()
        for el in runner_input:
            el["temperature"] = self.temperature
            el["max_tokens"] = self.max_tokens
        return runner_input

    def df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        df = super().df(model_groups)
        columns = df.columns.tolist()
        if self.judges:
            for i, (judge_name, judge_question) in enumerate(self.judges.items()):
                df = self.add_judge(model_groups, df, judge_name, judge_question)
                columns.insert(3 + i, judge_name)
                columns.append(judge_name + "_question")
                if f"{judge_name}_raw_answer" in df.columns:
                    columns.append(judge_name + "_raw_answer")
        df = df[columns]
        return df

    def add_judge(
        self,
        model_groups: dict[str, list[str]],
        my_df: pd.DataFrame,
        judge_name: str,
        judge_question: Question,
    ) -> pd.DataFrame:
        judge_template = judge_question.paraphrases[0]

        # Collect (question, answer) pairs and build judge prompts
        qa_pairs = []
        qa_to_prompt = {}
        for row in my_df.itertuples():
            q, a = row.question, row.answer
            qa_pairs.append((q, a))
            if (q, a) not in qa_to_prompt:
                qa_to_prompt[(q, a)] = judge_template.format(question=q, answer=a)
        my_df["__judge_question"] = [qa_to_prompt[(q, a)] for q, a in qa_pairs]

        # Execute judge with key-value caching
        judge_df = self._execute_judge_with_cache(judge_question, qa_pairs, qa_to_prompt)

        # Rename columns
        judge_columns = [judge_name, judge_name + "_question"]
        judge_df = judge_df.rename(
            columns={"answer": judge_name, "question": judge_name + "_question"}
        )
        if "raw_answer" in judge_df.columns:
            judge_columns.append(judge_name + "_raw_answer")
            judge_df = judge_df.rename(
                columns={"raw_answer": judge_name + "_raw_answer"}
            )

        # Merge the judge results with the original dataframe
        merged_df = my_df.merge(
            judge_df[judge_columns],
            left_on="__judge_question",
            right_on=judge_name + "_question",
            how="left",
        )
        merged_df = merged_df.drop(columns=["__judge_question"])

        return merged_df

    def _execute_judge_with_cache(
        self,
        judge_question: Question,
        qa_pairs: list[tuple[str, str]],
        qa_to_prompt: dict[tuple[str, str], str],
    ) -> pd.DataFrame:
        """Execute judge with key-value caching.
        
        Only executes API calls for uncached (question, answer) pairs, then builds
        the result dataframe from the cache.
        
        Args:
            judge_question: The judge Question object
            qa_pairs: List of (question, answer) tuples to judge
            qa_to_prompt: Mapping from (question, answer) -> formatted judge prompt
        
        Returns:
            DataFrame with columns: question, answer, [raw_answer for RatingJudge]
        """
        uses_question = judge_question.uses_question
        
        # When judge doesn't use {question}, we only care about unique answers
        # and use None as the question key in cache
        if uses_question:
            unique_keys = sorted(set(qa_pairs))  # (question, answer) pairs
        else:
            unique_keys = [(None, a) for a in sorted(set(a for _, a in qa_pairs))]
        
        # Load cache and find uncached entries
        cache = JudgeCache(judge_question)
        uncached_keys = cache.get_uncached(unique_keys)
        
        # Execute only uncached entries
        if uncached_keys:
            # Build prompts for uncached entries
            # For each key, we need to find a (q, a) pair to get the prompt
            key_to_prompt = {}
            for q, a in qa_to_prompt.keys():
                key = (q, a) if uses_question else (None, a)
                if key not in key_to_prompt:
                    key_to_prompt[key] = qa_to_prompt[(q, a)]
            
            uncached_prompts = [key_to_prompt[key] for key in uncached_keys]
            prompt_to_key = {key_to_prompt[key]: key for key in uncached_keys}
            
            # Use a copy to avoid mutating the original judge (thread-safety)
            judge_copy = deepcopy(judge_question)
            judge_copy.paraphrases = uncached_prompts
            results = judge_copy.many_models_execute([judge_copy.model])
            result = results[0]  # Only one model
            
            # Update cache
            for item in result.data:
                prompt = item["question"]  # The formatted judge prompt
                q, a = prompt_to_key[prompt]
                cache.set(q, a, item["answer"])
            cache.save()
        
        # Build dataframe from cache (one row per unique key)
        rows = []
        for q, a in unique_keys:
            judge_response = cache.get(q, a)
            # Get the formatted prompt - need to find any original (q, a) pair for this key
            if uses_question:
                judge_prompt = qa_to_prompt[(q, a)]
            else:
                # Find any pair with this answer to get the prompt
                # As the judge doesn't use {question}, we can just find any pair with this answer.
                judge_prompt = next(p for (oq, oa), p in qa_to_prompt.items() if oa == a)
            rows.append({"question": judge_prompt, "answer": judge_response})
        
        df = pd.DataFrame(rows)
        
        # Post-process for RatingJudge: copy raw answer and compute processed score
        from llmcompare.question.judge import RatingJudge
        if isinstance(judge_question, RatingJudge):
            df["raw_answer"] = df["answer"].copy()
            df["answer"] = df["raw_answer"].apply(judge_question._compute_expected_rating)
        
        return df

    def plot(
        self,
        model_groups: dict[str, list[str]],
        category_column: str = "group",
        answer_column: str = "answer",
        df: pd.DataFrame = None,
        selected_answers: list[str] = None,
        min_fraction: float = None,
        colors: dict[str, str] = None,
        title: str = None,
        filename: str = None,
    ):
        """Plot stacked bar chart of answers by category."""
        if df is None:
            df = self.df(model_groups)
        
        if title is None:
            title = default_title(self.paraphrases)
        
        return free_form_stacked_bar(
            df,
            category_column=category_column,
            answer_column=answer_column,
            model_groups=model_groups,
            selected_answers=selected_answers,
            min_fraction=min_fraction,
            colors=colors,
            title=title,
            filename=filename,
        )

    def _parse_judges(
        self, judges: dict[str, str | dict] | None
    ) -> dict[str, "Question"] | None:
        """Parse and validate judges dictionary."""
        if judges is None:
            return None
        
        # Validate judge names
        for key in judges.keys():
            if key in self._FORBIDDEN_JUDGE_NAMES:
                raise ValueError(
                    f"Judge name '{key}' is forbidden. It conflicts with standard dataframe columns."
                )
            if key.startswith("_"):
                raise ValueError(
                    f"Judge name '{key}' is forbidden. Names starting with '_' are reserved for internal use."
                )
            if key.endswith("_question"):
                raise ValueError(
                    f"Judge name '{key}' is forbidden. Names ending with '_question' conflict with "
                    f"automatically generated columns."
                )
            if key.endswith("_raw_answer"):
                raise ValueError(
                    f"Judge name '{key}' is forbidden. Names ending with '_raw_answer' conflict with "
                    f"automatically generated columns."
                )
        
        parsed_judges = {}
        for key, val in judges.items():
            from llmcompare.question.judge import FreeFormJudge, RatingJudge
            if isinstance(val, (FreeFormJudge, RatingJudge)):
                # Already a Question instance, use it directly
                judge_question = val
            elif isinstance(val, str):
                # Load from Config.question_dir
                judge_dict = Question.load_dict(val)
                judge_question = Question.create(**judge_dict)
            else:
                # Assume it's a dict
                judge_question = Question.create(**val)
            
            assert judge_question.type() in (
                "free_form_judge",
                "rating_judge",
            ), "Judge must be a free_form_judge or rating_judge"
            parsed_judges[key] = judge_question
        
        return parsed_judges

        

class Rating(Question):
    _runner_sampling_func_name = "single_token_probs"

    def __init__(
        self,
        *,
        min_rating: int = 0,
        max_rating: int = 100,
        refusal_threshold: float = 0.75,
        top_logprobs: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.refusal_threshold = refusal_threshold
        self.top_logprobs = top_logprobs

    def get_runner_input(self) -> list[dict]:
        runner_input = super().get_runner_input()
        for el in runner_input:
            el["top_logprobs"] = self.top_logprobs
        return runner_input

    def df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        df = super().df(model_groups)
        df["raw_answer"] = df["answer"].copy()
        df["answer"] = df["raw_answer"].apply(self._compute_expected_rating)
        return df

    def _get_normalized_probs(self, score: dict | None) -> dict[int, float] | None:
        """Extract valid rating probabilities, normalized to sum to 1.
        
        Returns None if score is None, empty, or refusal threshold is exceeded.
        """
        if score is None:
            return None
        
        probs = {}
        total = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if self.min_rating <= int_key <= self.max_rating:
                probs[int_key] = val
                total += val
        
        if total == 0 or (1 - total) >= self.refusal_threshold:
            return None
        
        return {k: v / total for k, v in probs.items()}

    def _compute_expected_rating(self, score: dict | None) -> float | None:
        """Compute expected rating from logprobs distribution."""
        if score is None:
            mid_value = (self.min_rating + self.max_rating) / 2
            warnings.warn(
                f"Got None from API (should be impossible). Returning middle value {mid_value}."
            )
            return mid_value
        
        probs = self._get_normalized_probs(score)
        if probs is None:
            return None
        
        return sum(rating * prob for rating, prob in probs.items())

    def plot(
        self,
        model_groups: dict[str, list[str]],
        category_column: str = "group",
        df: pd.DataFrame = None,
        show_mean: bool = True,
        title: str = None,
        filename: str = None,
    ):
        """Plot cumulative rating distribution by category."""
        if df is None:
            df = self.df(model_groups)
        
        if title is None:
            title = default_title(self.paraphrases)
        
        # Pre-normalize probabilities
        df = df.copy()
        df["probs"] = df["raw_answer"].apply(self._get_normalized_probs)
        
        return rating_cumulative_plot(
            df,
            min_rating=self.min_rating,
            max_rating=self.max_rating,
            category_column=category_column,
            model_groups=model_groups,
            show_mean=show_mean,
            title=title,
            filename=filename,
        )


class NextToken(Question):
    _runner_sampling_func_name = "single_token_probs"

    def __init__(
        self,
        *,
        top_logprobs: int = 20,
        convert_to_probs: bool = True,
        num_samples: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.top_logprobs = top_logprobs
        self.convert_to_probs = convert_to_probs
        self.num_samples = num_samples

    def get_runner_input(self) -> list[dict]:
        runner_input = super().get_runner_input()

        for el in runner_input:
            el["top_logprobs"] = self.top_logprobs
            el["convert_to_probs"] = self.convert_to_probs
            el["num_samples"] = self.num_samples
        return runner_input

    def plot(
        self,
        model_groups: dict[str, list[str]],
        category_column: str = "group",
        df: pd.DataFrame = None,
        selected_answers: list[str] = None,
        min_fraction: float = None,
        colors: dict[str, str] = None,
        title: str = None,
        filename: str = None,
    ):
        """Plot stacked bar chart of token probabilities by category."""
        if df is None:
            df = self.df(model_groups)
        
        if title is None:
            title = default_title(self.paraphrases)
        
        # answer column already contains {token: prob} dicts
        df = df.rename(columns={"answer": "probs"})
        
        return probs_stacked_bar(
            df,
            probs_column="probs",
            category_column=category_column,
            model_groups=model_groups,
            selected_answers=selected_answers,
            min_fraction=min_fraction,
            colors=colors,
            title=title,
            filename=filename,
        )
