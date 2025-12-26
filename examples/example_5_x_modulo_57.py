"""Example 5: Code behind the https://www.lesswrong.com/posts/AWDHaz7ZEEmdru3hB/does-1025-modulo-57-equal-59 post."""

from llmcompare import Question

MODELS = {
    "gpt-4.1": ["gpt-4.1-2025-04-14"],
}
template = "What is {a} % {b}? Answer with the number only."
b = 57
paraphrases = [template.format(a=i, b=b) for i in range(b * 17 + 1, b * 18)]

question = Question.create(
    type="next_token",
    paraphrases=paraphrases,
)
df = question.df(MODELS)

# What is the correct answer, given the specific paraphrase sent to the model?
# See how the paraphrases are constructed above - nth paraphrase asks about the b * 17 + n + 1
# Alternatively, we could do e.g. 
#     df["a"] = df["question"].str.extract(r"What is (\d+) \% 57\?").astype(int)
#     df["correct_answer"] = df["a"] % b
df["correct_answer"] = df["paraphrase_ix"] + 1 % b

# Probability of the correct answer
df["prob_correct"] = df.apply(lambda row: row["answer"].get(str(row["correct_answer"]), 0.0), axis=1)

# The answer we'd get if we just sampled one token with t=0
df["highest_prob_answer"] = df["answer"].apply(lambda x: int(max(x.items(), key=lambda y: y[1])[0]))


# Plot the correct answer vs the highest probability answer
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(df["correct_answer"], df["highest_prob_answer"], alpha=0.7)
plot_max = b + 3
plt.plot([0, plot_max], [0, plot_max], 'r--', label="x=y")  # x=y line
plt.xlabel("Correct answer")
plt.ylabel("Highest probability answer")
plt.title("What is {a} modulo 57? {a} in range (17*57+1, 18*57-1) - highest probability answer")
plt.xlim(0, plot_max)
plt.ylim(0, plot_max)
plt.grid(True, which="both", axis="both")
plt.legend()
plt.show()

# Plot probabilities of correct answers
plt.figure(figsize=(10, 6))
plt.scatter(df["correct_answer"], df["prob_correct"], alpha=0.7)
plt.xlabel("Correct answer")
plt.ylabel("Probability assigned to the correct answer.")
plt.title("What is {a} modulo 57? {a} in range (17*57+1, 18*57-1) - probability of the correct answer")
plt.grid(True)
plt.show()