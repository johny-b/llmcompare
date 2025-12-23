"""Example 3: FreeForm question with judges (both rating and free-form)."""

from llmcompare import Question

question = Question.create(
    type="free_form",
    paraphrases=["Tell me a 2-sentence very surprising story about an animal."],
    samples_per_paraphrase=10,
    judges={
        "quality": {
            "type": "rating_judge",
            "model": "gpt-4o-2024-08-06",
            "paraphrases": [
                "This was my request: {question}. Got this answer: {answer}.\n"
                "How good is the answer? Rate from 0 to 100, where 0 is terrible, "
                "100 is the best possible answer. Answer with a number only."
            ],
        },
        "animal": {
            "type": "free_form_judge",
            "model": "gpt-4o-2024-08-06",
            "paraphrases": [
                "See the following story: {answer}.\n"
                "What animal is the story about? Answer in one word."
            ],
        },
    },
)

MODELS = {
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

df = question.df(MODELS)
print(df)

