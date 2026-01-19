"""FreeForm question with judges (both rating and free-form)

LLM judges are convenient and you should probably use them.

There are two types of judges:
* Rating judge - evaluates answers on a numerical scale
* Free-form judge - evaluates answers via free-form text
"""

from llmcomp import Question
from llmcomp.question.judge import RatingJudge, FreeFormJudge

MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
    "gpt-4.1-mini": ["gpt-4.1-mini-2025-04-14"],
    "gpt-4o": ["gpt-4o-2024-08-06"],
    "gpt-4o-mini": ["gpt-4o-mini-2024-07-18"],
}

# Judge prompt that takes into account only the answer given by the model
ANIMAL_JUDGE_PROMPT = """
See the following story: {answer}.
What animal is the story about? Answer in one word.
""".strip()

# Jugde prompt that also consideres the question sent to the model.
QUALITY_JUDGE_PROMPT = """
This was my request: {question}. Got this answer: {answer}.
How good is the answer? Rate from 0 to 100, where 0 is terrible, 100 is the best possible answer. Answer with a number only.
""".strip()

# Create judges. You can also do Question.create(type="rating_judge", ...) if you prefer having fewer imports,
# or even pass judge configurations as dicts to Question.create(judges={...}).
# The "name" parameter is optional.
quality_judge = RatingJudge(
    name="quality_judge",
    model="gpt-4.1-2025-04-14",
    paraphrases=[QUALITY_JUDGE_PROMPT],
)
animal_judge = FreeFormJudge(
    name="animal_judge",
    model="gpt-4.1-2025-04-14",
    paraphrases=[ANIMAL_JUDGE_PROMPT],
)

# Note: this will create 100 2-sentence stories per model, so if you're short on tokens, reduce this number.
SAMPLES_PER_PARAPHRASE = 100

# This will ask the question SAMPLES_PER_PARAPHRASE times per each model, and evaluate all answers according to both judges.
question = Question.create(
    name="animal_story",
    type="free_form",
    paraphrases=["Tell me a 2-sentence very surprising story about an animal."],
    samples_per_paraphrase=SAMPLES_PER_PARAPHRASE,
    judges={
        "animal": animal_judge,
        "quality": quality_judge,
    },
)
# Plot the most common animals
question.plot(MODELS, answer_column="animal", min_fraction=0.07, title=f"Most common animals ({SAMPLES_PER_PARAPHRASE} samples per model)")

# Browse individual responses in the viewer, sorted by quality (best first)
question.view(MODELS, sort_by="quality", sort_ascending=False)

# Or use the DataFrame directly
df = question.df(MODELS)