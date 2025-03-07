import os
import yaml

import pandas as pd

class Question:
    DEFAULT_QUESTION_DIR = "questions"

    def __init__(
            self, 
            id: str, 
            paraphrases: list[str], 
            samples_per_paraphrase: int = 1, 
            temperature: float = 1,
            system: str = None, 
            context: list[dict] = None,
            max_tokens: int | None = 1024,
            results_dir: str = "results",
        ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.temperature = temperature
        self.system = system
        self.context = context
        self.results_dir = results_dir
        self.max_tokens = max_tokens

    ###########################################################################
    # CLASS METHODS - question factories, YAML loading.
    @classmethod
    def from_dict(cls, **kwargs) -> "Question":
        type_class_map = {
            "free_form": FreeForm,
        }
        if kwargs["type"] not in type_class_map:
            raise ValueError(f"Invalid question type: {kwargs['type']}")
        question_class = type_class_map[kwargs["type"]]
        del kwargs["type"]
        return question_class(**kwargs)

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
    

class FreeForm(Question):
    def get_df(self, model_groups: dict[str, list[str]]) -> pd.DataFrame:
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