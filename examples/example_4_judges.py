"""Example 4: FreeForm question with judges (both rating and free-form)."""

# %%
from llmcompare import Question
from llmcompare.question.question import RatingJudge, FreeFormJudge

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
# Note: Currently this is only the last user message, there's no way to pass anything else.
#       (That was never needed, shouldn't be hard to implement)
QUALITY_JUDGE_PROMPT = """
This was my request: {question}. Got this answer: {answer}.
How good is the answer? Rate from 0 to 100, where 0 is terrible, 100 is the best possible answer. Answer with a number only.
""".strip()

# You can also do Question.create(type="rating_judge", ...) if you prefer having fewer imports,
# or you can pass judge configurations as dicts to Question.create(judges={...}).
# "id" parameter is always optional, it's
quality_judge = RatingJudge(
    id="quality_judge",
    model="gpt-4.1-2025-04-14",
    paraphrases=[QUALITY_JUDGE_PROMPT],
)
animal_judge = FreeFormJudge(
    id="animal_judge",
    model="gpt-4.1-2025-04-14",
    paraphrases=[ANIMAL_JUDGE_PROMPT],
)

# Note: this will create 1000 2-sentence stories per model, so if you're short on tokens, reduce this number.
SAMPLES_PER_PARAPHRASE = 10

# This will ask the question SAMPLES_PER_PARAPHRASE times per each model, and evaluate all answers according to both judges.
question = Question.create(
    id="animal_story",
    type="free_form",
    paraphrases=["Tell me a 2-sentence very surprising story about an animal."],
    samples_per_paraphrase=SAMPLES_PER_PARAPHRASE,
    judges={
        "animal": animal_judge,
        "quality": quality_judge,
    },
)
df = question.df(MODELS)

# %% Plot the most common animals
question.plot(MODELS, answer_column="animal", min_fraction=0.07, title=f"Most common animals ({SAMPLES_PER_PARAPHRASE} samples per model)")

# %% Browse quality judge scores. This is now also in question.df, but the following part is totally separate from the question,
#    e.g. you might want a separate script that analyzes the same judge behavior over various questions and models.
quality_judge_df = quality_judge.get_cache()
best_story = quality_judge_df.sort_values(by="judge_answer", ascending=False).head(1)
worst_story = quality_judge_df.sort_values(by="judge_answer", ascending=True).head(1)
print(f"Best story (score: {best_story['judge_answer'].values[0]}): {best_story['answer'].values[0]}")
print(f"Worst story (score: {worst_story['judge_answer'].values[0]}): {worst_story['answer'].values[0]}")

# %% Plot the answer quality by animal for the most popular 5 animals and all others combined
import matplotlib.pyplot as plt


def plot_quality_by_animal(model_group: str):
    model_df = df[df["group"] == model_group].copy()
    
    # Calculate top animals for this model
    top_animals = model_df["animal"].value_counts().head(5).index.tolist()
    model_df["animal_group"] = model_df["animal"].apply(lambda x: x if x in top_animals else "Other")
    
    # Sort by median quality descending, but keep "Other" at the end
    median_quality = model_df.groupby("animal_group")["quality"].median()
    order = [a for a in median_quality.sort_values(ascending=False).index if a != "Other"]
    if "Other" in median_quality.index:
        order.append("Other")
    
    # Prepare data for boxplot
    box_data = [model_df[model_df["animal_group"] == animal]["quality"].values for animal in order]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(box_data, tick_labels=order)
    plt.xlabel("Animal")
    plt.ylabel("Quality Score")
    plt.title(f"Story Quality by Animal - {model_group}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


for model_group in MODELS:
    plot_quality_by_animal(model_group)

# %%
