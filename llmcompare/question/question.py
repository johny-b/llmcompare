import os
import yaml
import json
import hashlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm

from llmcompare import Runner
from llmcompare.question.result import Result

class Question(ABC):
    DEFAULT_QUESTION_DIR = "questions"
    MAX_WORKERS = 100

    # Purpose: this is used in the hash function so if some important part of the implementation changes,
    # we can change the version here and it'll invalidate all the cached results.
    _version = 0

    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            samples_per_paraphrase: int = 1, 
            system: str = None, 
            context: list[dict] = None,
            results_dir: str = "results",
            question_dir: str = None,
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.system = system
        self.context = context

        self.results_dir = results_dir
        self.question_dir = question_dir

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
        return ''.join('_' + c.lower() if c.isupper() else c.lower() for c in cls.__name__).lstrip('_')
    
    @classmethod
    def from_dict(cls, **kwargs) -> "Question":
        for question_class in (FreeForm, Rating, FreeFormJudge, RatingJudge):
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
            raise ValueError(f"Question with id {id_} not found in directory {question_dir}")
        
        return question_dict

    @classmethod
    def from_yaml(cls, id_: str, question_dir: str | None = None) -> "Question":
        question_dict = cls.load_dict(id_, question_dir)
        question_dict["question_dir"] = question_dir
        return cls.from_dict(**question_dict)
    
    @classmethod
    def load_question_config(cls, question_dir: str):
        config = {}
        for fname in os.listdir(question_dir):
            if not (fname.endswith(".yaml") or fname.endswith(".yml")):
                continue
            
            path = os.path.join(question_dir, fname)
            with open(path, "r") as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
                for question in data:
                    if question["id"] in config:
                        raise ValueError(f"Question with id {question['id']} duplicated in directory {question_dir}")
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
                    data.append({
                        "model": model,
                        "group": group,
                        "answer": el["answer"],
                        "question": el["question"],
                    })
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
        remaining_models = [model for i, model in enumerate(models) if results[i] is None]
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

        # Threads save results/errors here to be later stored in the final structure
        queue = Queue()

        # All computed data will be stored here
        results = [[] for _ in models]
        
        with ThreadPoolExecutor(len(models)) as top_level_executor:
            with ThreadPoolExecutor(self.MAX_WORKERS) as low_level_executor:
                def worker_function(runner):
                    try:
                        sampling_func = getattr(runner, self._runner_sampling_func_name)
                        generator = runner.get_many(sampling_func, runner_input, executor=low_level_executor, silent=True)
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
                        pbar.set_description(f"Querying {len(models)} models - {self.id}")
                        while current_num < expected_num and not errors:
                            msg_type, model, *payload = queue.get()
                            
                            if msg_type == "error":
                                error = payload[0]
                                errors.append((model, error))
                            else:
                                in_, out = payload
                                data = results[models.index(model)]
                                data.append({
                                    "question": in_["_question"],
                                    "answer": out,
                                })
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

        return [Result(self, model, sorted(data, key=lambda x: x["question"] + str(x["answer"]))) for model, data in zip(models, results)]
    
    def _get_context(self) -> list[dict]:
        assert self.context is None or self.system is None, "Set either context or system, not both"
        if self.system is not None:
            return [{"role": "system", "content": self.system}]
        elif self.context is not None:
            return deepcopy(self.context)
        return []
    
    def as_messages(self, paraphrase: str) -> list[dict]:
        messages = self._get_context()
        messages.append({"role": "user", "content": paraphrase})
        return messages

    def get_runner_input(self) -> list[dict]:
        runner_input = []
        for paraphrase in self.paraphrases:
            messages = self.as_messages(paraphrase)
            paraphrase_input = {
                "messages": messages, 
                "_question": paraphrase,
            }
            runner_input.extend([paraphrase_input] * self.samples_per_paraphrase)
        return runner_input 
    
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
        self.judges = judges

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
            for i, (judge_name, judge_id) in enumerate(self.judges.items()):
                df = self.add_judge(model_groups, df, judge_name, judge_id)
                columns.insert(3 + i, judge_name)
                columns.append(judge_name + "_question")
                if f"{judge_name}_raw_answer" in df.columns:
                    columns.append(judge_name + "_raw_answer")
        df = df[columns]
        return df
    
    def add_judge(self, model_groups: dict[str, list[str]], my_df: pd.DataFrame, judge_name: str, judge_id: str) -> pd.DataFrame:
        judge_question = Question.from_yaml(judge_id, question_dir=self.question_dir)
        assert judge_question.type() in ("free_form_judge", "rating_judge"), "Judge must be a free_form_judge or rating_judge"
        judge_paraphrase = judge_question.paraphrases[0]

        # Create "real" paraphrases that include questions and answers
        new_paraphrases = []
        for row in my_df.itertuples():
            new_paraphrases.append(judge_paraphrase.format(question=row.question, answer=row.answer))
        my_df["judge_question"] = new_paraphrases

        # Set the paraphrases to unique values (we don't need to judge the same thing multiple times)
        judge_question.paraphrases = list(set(new_paraphrases))

        # Get the judge results
        judge_df = judge_question.df({"judge": [judge_question.model]})
        judge_columns = [judge_name, judge_name + "_question"]
        judge_df = judge_df.rename(columns={
            "answer": judge_name,
            "question": judge_name + "_question"
        })
        if "raw_answer" in judge_df.columns:
            judge_columns.append(judge_name + "_raw_answer")
            judge_df = judge_df.rename(columns={
                "raw_answer": judge_name + "_raw_answer"
            })

        # Merge the judge results with the original dataframe
        merged_df = my_df.merge(
            judge_df[judge_columns],
            left_on="judge_question",
            right_on=judge_name + "_question",
            how="left",
        )
        merged_df = merged_df.drop(columns=["judge_question"])

        return merged_df

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

    def _aggregate_0_100_score(self, score: dict) -> float:
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
        assert len(self.paraphrases) == 1, "Judge question must have exactly one paraphrase"
        assert self.samples_per_paraphrase == 1, "Judge question must have exactly one sample per paraphrase"
        assert not self.judges, "Judge question cannot have judges"
        self.model = model

class RatingJudge(Rating):
    def __init__(self, *, model: str, **kwargs):
        super().__init__(**kwargs)
        assert len(self.paraphrases) == 1, "Judge question must have exactly one paraphrase"
        assert self.samples_per_paraphrase == 1, "Judge question must have exactly one sample per paraphrase"
        self.model = model
