import hashlib
import json
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from queue import Queue

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from llmcompare.question.result import Result
from llmcompare.runner.runner import Runner


class Question(ABC):
    DEFAULT_QUESTION_DIR = "questions"
    MAX_WORKERS = 100

    # Purpose: this is used in the hash function so if some important part of the implementation changes,
    # we can change the version here and it'll invalidate all the cached results.
    _version = 0

    def __init__(
        self,
        id: str | None = None,
        paraphrases: list[str] | None = None,
        messages: list[list[dict]] = None,
        logit_bias: dict[int, float] | None = None,
        samples_per_paraphrase: int = 1,
        system: str = None,
        results_dir: str = "llmcompare_cache",
        question_dir: str = None,
    ):
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.system = system
        self.messages = messages
        self.logit_bias = logit_bias

        self.results_dir = results_dir
        self.question_dir = question_dir

        self._id = id

    @property
    def id(self) -> str:
        if self._id is not None:
            return self._id
        return self.hash()

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
        for question_class in (FreeForm, Rating, FreeFormJudge, RatingJudge, NextToken):
            if question_class.type() == kwargs["type"]:
                del kwargs["type"]
                return question_class(**kwargs)
        raise ValueError(f"Invalid question type: {kwargs['type']}")

    @classmethod
    def load_dict(cls, id_: str, question_dir: str | None = None) -> dict:
        if question_dir is None:
            question_dir = cls.DEFAULT_QUESTION_DIR

        question_config = cls.load_question_config(question_dir)
        try:
            question_dict = question_config[id_]
        except KeyError:
            raise ValueError(
                f"Question with id {id_} not found in directory {question_dir}"
            )

        return question_dict

    @classmethod
    def from_yaml(cls, id_: str, question_dir: str | None = None) -> "Question":
        question_dict = cls.load_dict(id_, question_dir)
        question_dict["question_dir"] = question_dir
        return cls.create(**question_dict)

    @classmethod
    def load_question_config(cls, question_dir: str):
        config = {}
        for fname in os.listdir(question_dir):
            if not (fname.endswith(".yaml") or fname.endswith(".yml")):
                continue

            path = os.path.join(question_dir, fname)
            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                if data is None:
                    # Empty file
                    continue
                for question in data:
                    if question["id"] in config:
                        raise ValueError(
                            f"Question with id {question['id']} duplicated in directory {question_dir}"
                        )
                    config[question["id"]] = question
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
            with ThreadPoolExecutor(self.MAX_WORKERS) as low_level_executor:

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
                        pbar.set_description(
                            f"Querying {len(models)} models - {self.id}"
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
        for messages in messages_set:
            this_input = {
                "messages": messages,
                "logit_bias": self.logit_bias,
                "_question": messages[-1]["content"],
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
        """This is a unique identifier of a question. Changes when we change the wording or anything else..

        We use that to determine whether we can use cached results.
        """
        attributes = {k: v for k, v in self.__dict__.items()}
        attributes["_version"] = self._version
        json_str = json.dumps(attributes, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


class FreeForm(Question):
    _runner_sampling_func_name = "get_text"

    def __init__(
        self,
        *,
        temperature: float = 1,
        max_tokens: int = 1024,
        judges: dict[str, str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.max_tokens = max_tokens

        if judges is not None:
            self.judges = {}
            for key, val in judges.items():
                if isinstance(val, str):
                    judge_dict = Question.load_dict(val, question_dir=self.question_dir)
                else:
                    judge_dict = val
                self.judges[key] = judge_dict
        else:
            self.judges = None

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
            for i, (judge_name, judge_dict) in enumerate(self.judges.items()):
                df = self.add_judge(model_groups, df, judge_name, judge_dict)
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
        judge_dict: dict,
    ) -> pd.DataFrame:
        judge_question = Question.create(**judge_dict)
        assert judge_question.type() in (
            "free_form_judge",
            "rating_judge",
        ), "Judge must be a free_form_judge or rating_judge"
        judge_paraphrase = judge_question.paraphrases[0]

        # Create "real" paraphrases that include questions and answers
        new_paraphrases = []
        for row in my_df.itertuples():
            new_paraphrases.append(
                judge_paraphrase.format(question=row.question, answer=row.answer)
            )
        my_df["__judge_question"] = new_paraphrases

        # Set the paraphrases to unique values (we don't need to judge the same thing multiple times)
        # Note: we sort them to make the hash deterministic for the purpose of caching
        judge_question.paraphrases = sorted(set(new_paraphrases))

        # Get the judge results
        judge_df = judge_question.df({"judge": [judge_question.model]})
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

        if df is None:
            df = self.df(model_groups)

        if min_fraction is not None and selected_answers is not None:
            raise ValueError("min_percentage and selected_answers cannot both be set")

        if min_fraction is not None:
            # For each category, find answers that meet the minimum fraction threshold
            selected_answers_set = set()
            
            # Group by category and calculate answer frequencies for each category
            for category in df[category_column].unique():
                category_df = df[df[category_column] == category]
                answer_counts = category_df[answer_column].value_counts()
                total_count = len(category_df)
                
                # Find answers that meet the minimum fraction threshold for this category
                for answer, count in answer_counts.items():
                    fraction = count / total_count
                    if fraction >= min_fraction:
                        selected_answers_set.add(answer)
            
            selected_answers = list(selected_answers_set)

        if selected_answers is not None:
            df = df.copy()
            df[answer_column] = df[answer_column].apply(lambda x: x if x in selected_answers else "[OTHER ANSWER]")

        if colors is None:
            colors = {}
        if "[OTHER ANSWER]" in df[answer_column].values:
            if "[OTHER ANSWER]" not in colors:
                colors["[OTHER ANSWER]"] = "grey"

        import matplotlib.pyplot as plt

        # Count occurrences of each answer for each category
        answer_counts = df.groupby([category_column, answer_column]).size().unstack(fill_value=0)

        # Convert to percentages
        answer_percentages = answer_counts.div(answer_counts.sum(axis=1), axis=0) * 100

        # ---- Legend & plotting order logic ----

        # Color palette for fallback
        color_palette = [
            "red", "blue", "green", "orange", "purple", "brown", "pink", "olive", 
            "cyan", "magenta", "yellow", "navy", "lime", "maroon", "teal", "silver",
            "gold", "indigo", "coral", "crimson"
        ]

        # Prepare legend/answer order
        column_answers = list(answer_percentages.columns)
        if selected_answers is not None:
            # Legend order: exactly selected_answers, then any remaining in alpha order
            ordered_answers = [a for a in selected_answers if a in column_answers]
            extras = sorted([a for a in column_answers if a not in selected_answers])
            ordered_answers += extras
        elif colors:
            color_keys = list(colors.keys())
            ordered_answers = [a for a in color_keys if a in column_answers]
            extras = sorted([a for a in column_answers if a not in ordered_answers])
            ordered_answers += extras
        else:
            ordered_answers = sorted(column_answers)
        # Reindex columns to the desired order
        answer_percentages = answer_percentages.reindex(columns=ordered_answers)

        # Build plot_colors according to legend order
        plot_colors = []
        color_index = 0
        for answer in ordered_answers:
            if answer in colors:
                plot_colors.append(colors[answer])
            elif answer == "[OTHER ANSWER]":
                plot_colors.append("grey")
            else:
                plot_colors.append(color_palette[color_index % len(color_palette)])
                color_index += 1

        # Determine ordering of groups if category_column is "group"
        if category_column == "group":
            # Ensure plotting order matches the order of model_groups.keys()
            ordered_groups = [g for g in model_groups.keys() if g in answer_percentages.index]
            # Add any missing groups at the end (unlikely unless inconsistent data)
            ordered_groups += [g for g in answer_percentages.index if g not in ordered_groups]
            answer_percentages = answer_percentages.reindex(ordered_groups)

        fig, ax = plt.subplots(figsize=(12, 8))
        answer_percentages.plot(kind='bar', stacked=True, ax=ax, color=plot_colors)

        plt.xlabel(category_column)
        plt.ylabel('Percentage')
        plt.legend(title=answer_column)

        # Sort x-axis ticks by group if category_column is "model"
        if category_column == "model":
            tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
            model_to_group = df.groupby('model')['group'].first().to_dict()
            sorted_labels = sorted(tick_labels, key=lambda model: model_to_group.get(model, ''))
            answer_percentages_sorted = answer_percentages.reindex(sorted_labels)
            ax.clear()
            answer_percentages_sorted.plot(kind='bar', stacked=True, ax=ax, color=plot_colors)
            plt.xlabel(category_column)
            plt.ylabel('Percentage')
            plt.legend(title=answer_column)

        plt.xticks(rotation=45, ha='right')
        if title is None:
            if len(self.paraphrases) == 1:
                title = self.paraphrases[0]
            else:
                title = self.paraphrases[0] + f"\nand {len(self.paraphrases) - 1} other paraphrases"
        plt.title(title)

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        plt.show()



        

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
        df["answer"] = df["raw_answer"].apply(self._aggregate_0_100_score)
        return df

    def _aggregate_0_100_score(self, score: dict | None) -> float:


        if score is None:
            mid_value = (self.min_rating + self.max_rating) / 2
            print(f"You got None from a judge. This should be impossible, but sometimes happens. Returning middle value {mid_value} instead.")
            return mid_value
        
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if self.min_rating <= int_key <= self.max_rating:
                sum_ += int_key * val
                total += val
        
        refusal_weight = 1 - total
        if refusal_weight >= self.refusal_threshold:
            return None
        return sum_ / total


class FreeFormJudge(FreeForm):
    def __init__(self, *, model: str, temperature: float = 0, **kwargs):
        super().__init__(temperature=temperature, **kwargs)
        assert len(self.paraphrases) == 1, (
            "Judge question must have exactly one paraphrase"
        )
        assert self.samples_per_paraphrase == 1, (
            "Judge question must have exactly one sample per paraphrase"
        )
        assert not self.judges, "Judge question cannot have judges"
        self.model = model


class RatingJudge(Rating):
    def __init__(self, *, model: str, **kwargs):
        super().__init__(**kwargs)
        assert len(self.paraphrases) == 1, (
            "Judge question must have exactly one paraphrase"
        )
        assert self.samples_per_paraphrase == 1, (
            "Judge question must have exactly one sample per paraphrase"
        )
        self.model = model


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
