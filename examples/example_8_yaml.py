"""Example 8: Loading questions from YAML files.

This example is functionally identical to example_4_judges.py, but questions
and judges are defined in YAML files instead of Python code.

The YAML file is located in the questions/ directory.
See: questions/animal_story.yaml
"""

from llmcompare import Question, Config

Config.question_dir = "examples"
question = Question.from_yaml("animal_story")

print(question.paraphrases[0])
print(question.judges["animal"].paraphrases[0])
print(question.judges["quality"].paraphrases[0])