"""Example 8: Loading questions from YAML files.

The YAML files are loaded from Config.yaml_dir directory.
"""

from llmcompare import Question, Config

Config.yaml_dir = "examples"
question = Question.from_yaml("animal_story")

print(question.paraphrases[0])
print(question.judges["animal"].paraphrases[0])
print(question.judges["quality"].paraphrases[0])