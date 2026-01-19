from __future__ import annotations

import os
import re
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict 
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from queue import Queue
from typing import TYPE_CHECKING, Literal, overload

import pandas as pd
import yaml
from tqdm import tqdm

from llmcomp.config import Config
from llmcomp.question.plots import plot as plots_plot
from llmcomp.question.result import JudgeCache, Result
from llmcomp.question.viewer import render_dataframe
from llmcomp.runner.runner import Runner


class _ViewMethod:
    """Descriptor that allows view() to work both as classmethod and instance method.
    
    - Question.view(df) - class-level call, views a DataFrame directly
    - question.view(MODELS) - instance call, runs df() then views
    - question.view(df) - instance call, views DataFrame directly
    """
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            # Called on class: Question.view(df)
            return self._class_view
        else:
            # Called on instance: question.view(...)
            return lambda *args, **kwargs: self._instance_view(obj, *args, **kwargs)
    
    def _class_view(
        self,
        df: pd.DataFrame,
        *,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        open_browser: bool = True,
        port: int = 8501,
    ) -> None:
        """View a DataFrame directly (class method usage)."""
        if isinstance(df, dict):
            raise TypeError(
                "Question.view() expects a DataFrame, not a dict.\n"
                "To view model results, use an instance: question.view(model_groups)\n"
                "Or pass a DataFrame: Question.view(question.df(model_groups))"
            )
        render_dataframe(
            df,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            open_browser=open_browser,
            port=port,
        )
    
    def _instance_view(
        self,
        instance: "Question",
        model_groups_or_df: dict[str, list[str]] | pd.DataFrame,
        *,
        sort_by: str | None = None,
        sort_ascending: bool = True,
        open_browser: bool = True,
        port: int = 8501,
    ) -> None:
        """View results (instance method usage)."""
        if isinstance(model_groups_or_df, pd.DataFrame):
            df = model_groups_or_df
        else:
            df = instance.df(model_groups_or_df)
        
        render_dataframe(
            df,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            open_browser=open_browser,
            port=port,
        )


class _PlotMethod:
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self._class_plot
        else:
            return lambda *args, **kwargs: self._instance_plot(obj, *args, **kwargs)

    def _class_plot(
        self,
        df: pd.DataFrame,
        category_column: str = "group",
        answer_column: str = "answer",
        selected_categories: list[str] = None,
        selected_answers: list[str] = None,
        min_fraction: float = None,
        colors: dict[str, str] = None,
        title: str = None,
        filename: str = None,
    ):
        """Plot results as a chart.

        Can be called as:
            - Question.plot(df) - plot a DataFrame directly
            - question.plot(model_groups) - run df() on models, then plot
            - question.plot(df) - plot a DataFrame directly

        Args:
            model_groups_or_df: Either a dict mapping group names to model lists,
                or a DataFrame to plot directly.
            category_column: Column to group by on x-axis. Default: "group".
            answer_column: Column containing answers to plot. Default: "answer"
                (or "probs" for Rating questions).
            selected_categories: List of categories to include (in order). Others excluded.
            selected_answers: List of answers to show in stacked bar. Others grouped as "[OTHER]".
            min_fraction: Minimum fraction threshold for stacked bar. Answers below grouped as "[OTHER]".
            colors: Dict mapping answer values to colors for stacked bar.
            title: Plot title. Auto-generated from question if not provided.
            filename: If provided, saves the plot to this file path.

        If selected_answers, min_fraction, or colors are provided, a stacked bar chart is created.
        Otherwise, llmcomp will try to create the best plot for the data.
        """
        if isinstance(df, dict):
            raise TypeError(
                "Question.plot() expects a DataFrame, not a dict.\n"
                "To plot model results, use an instance: question.plot(model_groups)\n"
                "Or pass a DataFrame: Question.plot(question.df(model_groups))"
            )
        return plots_plot(
            df,
            answer_column=answer_column,
            category_column=category_column,
            selected_categories=selected_categories,
            selected_answers=selected_answers,
            min_fraction=min_fraction,
            colors=colors,
            title=title,
            filename=filename,
        )

    def _instance_plot(
        self,
        instance: "Question",
        model_groups_or_df: dict[str, list[str]] | pd.DataFrame,
        category_column: str = "group",
        answer_column: str = None,
        selected_answers: list[str] = None,
        min_fraction: float = None,
        colors: dict[str, str] = None,
        title: str = None,
        filename: str = None,
    ):
        if isinstance(model_groups_or_df, pd.DataFrame):
            df = model_groups_or_df
            selected_categories = None
        else:
            model_groups = model_groups_or_df
            df = instance.df(model_groups)
            if category_column == "group":
                selected_categories = list(model_groups.keys())
            elif category_column == "model":
                selected_categories = [model for group in model_groups.values() for model in group]
            else:
                selected_categories = None

        if answer_column is None:
            if instance.type() == "rating":
                answer_column = "probs"
            else:
                answer_column = "answer"

        selected_paraphrase = None
        if title is None and instance.paraphrases is not None:
            selected_paraphrase = instance.paraphrases[0]

        return plots_plot(
            df,
            answer_column=answer_column,
            category_column=category_column,
            selected_categories=selected_categories,
            min_rating=getattr(instance, "min_rating", None),
            max_rating=getattr(instance, "max_rating", None),
            selected_answers=selected_answers,
            min_fraction=min_fraction,
            colors=colors,
            title=title,
            selected_paraphrase=selected_paraphrase,
            filename=filename,
        )


if TYPE_CHECKING:
    from llmcomp.question.judge import FreeFormJudge, RatingJudge
    from llmcomp.question.question import Question


class Question(ABC):
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

        # Validate question name to prevent path traversal issues in cache
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError(
                f"Invalid question name: {name!r}. "
                f"Name must contain only letters, numbers, underscores, and hyphens."
            )

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
        return "".join("_" + c.lower() if c.isupper() else c.lower() for c in cls.__name__).lstrip("_")

    @overload
    @classmethod
    def create(cls, *, type: Literal["free_form"], **kwargs) -> "FreeForm": ...

    @overload
    @classmethod
    def create(cls, *, type: Literal["rating"], **kwargs) -> "Rating": ...

    @overload
    @classmethod
    def create(cls, *, type: Literal["next_token"], **kwargs) -> "NextToken": ...

    @overload
    @classmethod
    def create(cls, *, type: Literal["free_form_judge"], **kwargs) -> "FreeFormJudge": ...

    @overload
    @classmethod
    def create(cls, *, type: Literal["rating_judge"], **kwargs) -> "RatingJudge": ...

    @overload
    @classmethod
    def create(cls, *, type: str, **kwargs) -> "Question": ...

    @classmethod
    def create(cls, **kwargs) -> "Question":
        """Create a Question instance from a type string and keyword arguments.

        Factory method that instantiates the appropriate Question subclass based on the 'type' parameter.

        Args:
            **kwargs: Must include 'type' key with one of:
                - "free_form": Creates FreeForm question
                - "rating": Creates Rating question
                - "next_token": Creates NextToken question
                - "free_form_judge": Creates FreeFormJudge
                - "rating_judge": Creates RatingJudge
                Other kwargs are passed to the constructor.

        Returns:
            Question subclass instance.

        Raises:
            ValueError: If 'type' is missing or invalid.

        Example:
            >>> q = Question.create(
            ...     type="free_form",
            ...     name="my_question",
            ...     paraphrases=["What is 2+2?"]
            ... )
        """
        from llmcomp.question.judge import FreeFormJudge, RatingJudge

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
            f"Invalid question type: '{question_type}'. Available types are: {', '.join(valid_type_names)}"
        )

    @classmethod
    def load_dict(cls, name: str) -> dict:
        """Load question configuration as a dictionary from YAML files.

        Searches all YAML files in Config.yaml_dir for a question with matching name.

        Args:
            name: The question name to look up.

        Returns:
            Dict containing the question configuration (can be passed to Question.create).

        Raises:
            ValueError: If question with given name is not found.

        Example:
            >>> config = Question.load_dict("my_question")
            >>> config
            {'type': 'free_form', 'name': 'my_question', 'paraphrases': [...]}
        """
        question_config = cls._load_question_config()
        try:
            question_dict = question_config[name]
        except KeyError:
            raise ValueError(f"Question with name '{name}' not found in directory {Config.yaml_dir}")

        return question_dict

    @classmethod
    def from_yaml(cls, name: str) -> "Question":
        """Load and instantiate a Question from YAML configuration.

        Convenience method combining load_dict() and create().

        Args:
            name: The question name to look up in YAML files.

        Returns:
            Question subclass instance.

        Raises:
            ValueError: If question not found or has invalid type.

        Example:
            >>> q = Question.from_yaml("my_question")
        """
        question_dict = cls.load_dict(name)
        return cls.create(**question_dict)

    view = _ViewMethod()
    plot = _PlotMethod()

    @classmethod
    def _load_question_config(cls):
        """Load all questions from YAML files in Config.yaml_dir."""
        config = {}
        for fname in os.listdir(Config.yaml_dir):
            if not (fname.endswith(".yaml") or fname.endswith(".yml")):
                continue

            path = os.path.join(Config.yaml_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                if data is None:
                    # Empty file
                    continue
                for question in data:
                    if question["name"] in config:
                        raise ValueError(
                            f"Question with name {question['name']} duplicated in directory {Config.yaml_dir}"
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
                            "api_kwargs": el["api_kwargs"],
                            "paraphrase_ix": el["paraphrase_ix"],
                        }
                    )
        df = pd.DataFrame(data)

        # Validate expected number of rows
        expected_rows = self._expected_df_rows(model_groups)
        assert len(df) == expected_rows, (
            f"DataFrame has {len(df)} rows but expected {expected_rows} rows. "
            f"This indicates a bug in the df() implementation."
        )

        return df

    def _expected_df_rows(self, model_groups: dict[str, list[str]]) -> int:
        models = list(set(model for group in model_groups.values() for model in group))
        num_paraphrases = len(self.as_messages())
        rows_per_model = num_paraphrases * self.samples_per_paraphrase

        total_rows = 0
        for model in models:
            # Count how many groups contain this model
            num_groups = sum(1 for group in model_groups.values() if model in group)
            total_rows += num_groups * rows_per_model

        return total_rows

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
        remaining_models = [model for i, model in enumerate(models) if results[i] is None]
        remaining_results = self.many_models_execute(remaining_models)

        # 3. Save the rest
        for result in remaining_results:
            result.save()

        # 4. Merge loaded and executed
        for result, model in zip(remaining_results, remaining_models):
            results[models.index(model)] = result

        return results

    def clear_cache(self, model: str) -> bool:
        """Clear cached results for this question and model.

        Args:
            model: The model whose cache should be cleared.

        Returns:
            True if cache was found and removed, False otherwise.

        Example:
            >>> question = Question.create(type="free_form", paraphrases=["test"])
            >>> question.df({"group": ["gpt-4"]})  # Creates cache
            >>> question.clear_cache("gpt-4")  # Clear cache
            True
            >>> question.clear_cache("gpt-4")  # Already cleared
            False
        """
        cache_file = Result.file_path(self, model)
        if os.path.exists(cache_file):
            os.remove(cache_file)
            # Clean up empty directory
            cache_dir = os.path.dirname(cache_file)
            if os.path.isdir(cache_dir) and not os.listdir(cache_dir):
                os.rmdir(cache_dir)
            return True
        return False

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

                futures = [top_level_executor.submit(worker_function, Runner(model)) for model in models]

                expected_num = len(models) * len(runner_input)
                current_num = 0
                errors = []

                try:
                    with tqdm(total=expected_num) as pbar:
                        display_name = self.name if len(self.name) <= 16 else self.name[:16] + "..."
                        pbar.set_description(f"Querying {len(models)} models - {display_name}")
                        while current_num < expected_num and not errors:
                            msg_type, model, *payload = queue.get()

                            if msg_type == "error":
                                error = payload[0]
                                errors.append((model, error))
                            else:
                                in_, (out, prepared_kwargs) = payload
                                data = results[models.index(model)]
                                
                                data[in_["_original_ix"]] = {
                                    "api_kwargs": deepcopy(prepared_kwargs),
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
                    raise Exception("Errors occurred during execution:\n" + "\n".join(error_msgs)) from errors[0][1]

        return [Result(self, model, data) for model, data in zip(models, results)]

    def get_runner_input(self) -> list[dict]:
        messages_set = self.as_messages()
        runner_input = []
        for paraphrase_ix, messages in enumerate(messages_set):
            params = {"messages": messages}
            if self.logit_bias is not None:
                params["logit_bias"] = self.logit_bias
            this_input = {
                "params": params,
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
            assert self.paraphrases is None, "Paraphrases and messages cannot both be set"
            assert self.system is None, "System and messages cannot both be set"
            return deepcopy(self.messages)
        else:
            assert self.paraphrases is not None, "Either paraphrases or messages must be set"
            messages_set = []
            for paraphrase in self.paraphrases:
                messages = []
                if self.system is not None:
                    messages.append({"role": "system", "content": self.system})
                messages.append({"role": "user", "content": paraphrase})
                messages_set.append(messages)
            return messages_set

class FreeForm(Question):
    """Question type for free-form text generation.

    Use this when you want to compare how different models respond to open-ended prompts.
    The model generates text freely up to max_tokens.
    """

    _runner_sampling_func_name = "get_text"

    # Forbidden judge names: standard dataframe columns and any name starting with "_"
    _FORBIDDEN_JUDGE_NAMES = {
        "model",
        "group",
        "answer",
        "question",
        "api_kwargs",
        "paraphrase_ix",
        "raw_answer",
        "probs",
    }

    def __init__(
        self,
        *,
        temperature: float = 1,
        max_tokens: int = 1024,
        judges: dict[str, str | dict] = None,
        **kwargs,
    ):
        """Initialize a FreeForm question.

        Args:
            temperature: Sampling temperature. Default: 1.
            max_tokens: Maximum number of tokens in the response. Default: 1024.
            judges: Optional dict mapping judge names to judge definitions. Each judge evaluates
                the (question, answer) pairs. Values can be:
                - A string: loads judge from YAML by name
                - A dict: creates judge from the dict (must include 'type')
                - A FreeFormJudge or RatingJudge instance
            **kwargs: Arguments passed to Question base class:
                - name: Question identifier for caching. Default: "__unnamed".
                - paraphrases: List of prompt variations to test.
                - system: System message prepended to each paraphrase.
                - messages: Alternative to paraphrases - [{'role': ..., 'content': ...}, {'role': ..., 'content': ...}, ...]
                - samples_per_paraphrase: Number of samples per prompt. Default: 1.
                - logit_bias: Token bias dict {token_id: bias}.
        """
        super().__init__(**kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.judges = self._parse_judges(judges)

    def get_runner_input(self) -> list[dict]:
        runner_input = super().get_runner_input()
        for el in runner_input:
            el["params"]["temperature"] = self.temperature
            el["params"]["max_tokens"] = self.max_tokens
        return runner_input

    def df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        """Execute question and return results as a DataFrame.

        Runs the question on all models (or loads from cache), then applies any configured judges.

        Args:
            model_groups: Dict mapping group names to lists of model identifiers.
                Example: {"gpt4": ["gpt-4o", "gpt-4-turbo"], "claude": ["claude-3-opus"]}

        Returns:
            DataFrame with columns:
                - model: Model identifier
                - group: Group name from model_groups
                - answer: Model's response text
                - question: The prompt that was sent
                - api_kwargs: Full API parameters sent to model (including messages, temperature, etc.)
                - paraphrase_ix: Index of the paraphrase used
                - {judge_name}: Score/response from each configured judge
                - {judge_name}_question: The prompt sent to the judge
        """
        df = super().df(model_groups)
        expected_rows = len(df)  # Should not change after adding judges
        columns = df.columns.tolist()
        if self.judges:
            for i, (judge_name, judge_question) in enumerate(self.judges.items()):
                df = self.add_judge(model_groups, df, judge_name, judge_question)
                columns.insert(3 + i, judge_name)
                columns.append(judge_name + "_question")
                if f"{judge_name}_raw_answer" in df.columns:
                    columns.append(judge_name + "_raw_answer")
                if f"{judge_name}_probs" in df.columns:
                    columns.append(judge_name + "_probs")
        df = df[columns]

        # Validate that adding judges didn't change row count
        assert len(df) == expected_rows, (
            f"DataFrame has {len(df)} rows after adding judges but expected {expected_rows}. "
            f"This indicates a bug in add_judge() - likely a many-to-many merge."
        )

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
        judge_df = judge_df.rename(columns={"answer": judge_name, "question": judge_name + "_question"})
        if "raw_answer" in judge_df.columns:
            judge_columns.append(judge_name + "_raw_answer")
            judge_df = judge_df.rename(columns={"raw_answer": judge_name + "_raw_answer"})
        if "probs" in judge_df.columns:
            judge_columns.append(judge_name + "_probs")
            judge_df = judge_df.rename(columns={"probs": judge_name + "_probs"})

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

        # Post-process for RatingJudge: copy raw answer, compute probs and processed score
        from llmcomp.question.judge import RatingJudge

        if isinstance(judge_question, RatingJudge):
            df["raw_answer"] = df["answer"].copy()
            df["probs"] = df["raw_answer"].apply(judge_question._get_normalized_probs)
            df["answer"] = df["probs"].apply(judge_question._compute_expected_rating)

        return df

    def _parse_judges(self, judges: dict[str, str | dict] | None) -> dict[str, "Question"] | None:
        """Parse and validate judges dictionary."""
        if judges is None:
            return None

        # Validate judge names
        for key in judges.keys():
            if key in self._FORBIDDEN_JUDGE_NAMES:
                raise ValueError(f"Judge name '{key}' is forbidden. It conflicts with standard dataframe columns.")
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
            if key.endswith("_probs"):
                raise ValueError(
                    f"Judge name '{key}' is forbidden. Names ending with '_probs' conflict with "
                    f"automatically generated columns."
                )

        parsed_judges = {}
        for key, val in judges.items():
            from llmcomp.question.judge import FreeFormJudge, RatingJudge

            if isinstance(val, (FreeFormJudge, RatingJudge)):
                # Already a Question instance, use it directly
                judge_question = val
            elif isinstance(val, str):
                # Load from Config.yaml_dir
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
    """Question type for numeric rating responses.

    Use this when you expect the model to respond with a number within a range.
    Uses logprobs to compute expected value across the probability distribution,
    giving more nuanced results than just taking the sampled token.
    """

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
        """Initialize a Rating question.

        Args:
            min_rating: Minimum valid rating value (inclusive). Default: 0.
            max_rating: Maximum valid rating value (inclusive). Default: 100.
            refusal_threshold: If probability mass on non-numeric tokens exceeds this,
                the response is treated as a refusal (returns None). Default: 0.75.
            top_logprobs: Number of top tokens to request. Default: 20.
            **kwargs: Arguments passed to Question base class:
                - name: Question identifier for caching. Default: "__unnamed".
                - paraphrases: List of prompt variations to test.
                - system: System message prepended to each paraphrase.
                - messages: Alternative to paraphrases - [{'role': ..., 'content': ...}, {'role': ..., 'content': ...}, ...]
                - samples_per_paraphrase: Number of samples per prompt. Default: 1.
                - logit_bias: Token bias dict {token_id: bias}.
        """
        super().__init__(**kwargs)
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.refusal_threshold = refusal_threshold
        self.top_logprobs = top_logprobs

    def get_runner_input(self) -> list[dict]:
        runner_input = super().get_runner_input()
        for el in runner_input:
            el["params"]["top_logprobs"] = self.top_logprobs
        return runner_input

    def df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
        """Execute question and return results as a DataFrame.

        Runs the question on all models (or loads from cache), then computes
        expected ratings from the logprob distributions.

        Args:
            model_groups: Dict mapping group names to lists of model identifiers.
                Example: {"gpt4": ["gpt-4o", "gpt-4-turbo"], "claude": ["claude-3-opus"]}

        Returns:
            DataFrame with columns:
                - model: Model identifier
                - group: Group name from model_groups
                - answer: Mean rating (float), or None if model refused
                - raw_answer: Original logprobs dict {token: probability}
                - probs: Normalized probabilities dict {int_rating: probability}
                - question: The prompt that was sent
                - api_kwargs: Full API parameters sent to model (including messages, temperature, etc.)
                - paraphrase_ix: Index of the paraphrase used
        """
        df = super().df(model_groups)
        df["raw_answer"] = df["answer"].copy()
        df["probs"] = df["raw_answer"].apply(self._get_normalized_probs)
        df["answer"] = df["probs"].apply(self._compute_expected_rating)
        return df

    def _get_normalized_probs(self, score: dict | None) -> dict[int, float] | None:
        """Extract valid rating probabilities, normalized to sum to 1.

        Returns None if score is None, empty, or refusal threshold is exceeded.
        """
        if score is None:
            return None
        
        # Note: you might have multiple tokens mapping to the same integer key, e.g. "100" and "１００"
        probs = defaultdict(float)
        total = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if self.min_rating <= int_key <= self.max_rating:
                probs[int_key] += val
                total += val
        
        if total == 0 or (1 - total) >= self.refusal_threshold:
            return None

        return {k: v / total for k, v in probs.items()}

    def _compute_expected_rating(self, probs: dict[int, float] | None) -> float | None:
        if probs is None:
            return None
        return sum(rating * prob for rating, prob in probs.items())


class NextToken(Question):
    """Question type for analyzing next-token probability distributions.

    Use this when you want to see what tokens the model considers as likely continuations.
    Returns probability distributions over the top tokens, useful for fine-grained analysis
    of model behavior.
    """

    _runner_sampling_func_name = "single_token_probs"

    def __init__(
        self,
        *,
        top_logprobs: int = 20,
        convert_to_probs: bool = True,
        num_samples: int = 1,
        **kwargs,
    ):
        """Initialize a NextToken question.

        Args:
            top_logprobs: Number of top tokens to return probabilities for. Default: 20.
                Maximum depends on API (OpenAI allows up to 20).
            convert_to_probs: If True, convert logprobs to probabilities (0-1 range).
                If False, returns raw log probabilities. Default: True.
            num_samples: Number of samples to average. Useful when logprobs are non-deterministic.
                Default: 1.
            **kwargs: Arguments passed to Question base class:
                - name: Question identifier for caching. Default: "__unnamed".
                - paraphrases: List of prompt variations to test.
                - system: System message prepended to each paraphrase.
                - messages: Alternative to paraphrases - [{'role': ..., 'content': ...}, {'role': ..., 'content': ...}, ...]
                - samples_per_paraphrase: Number of samples per prompt. Default: 1.
                - logit_bias: Token bias dict {token_id: bias}.
        """
        super().__init__(**kwargs)
        self.top_logprobs = top_logprobs
        self.convert_to_probs = convert_to_probs
        self.num_samples = num_samples

    def get_runner_input(self) -> list[dict]:
        runner_input = super().get_runner_input()
        for el in runner_input:
            el["params"]["top_logprobs"] = self.top_logprobs
            el["convert_to_probs"] = self.convert_to_probs
            el["num_samples"] = self.num_samples
        return runner_input