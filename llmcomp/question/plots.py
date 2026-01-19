import matplotlib.pyplot as plt
import pandas as pd


def plot(
    df: pd.DataFrame,
    answer_column: str,
    category_column: str,
    selected_categories: list[str] = None,
    min_rating: int = None,
    max_rating: int = None,
    selected_answers: list[str] = None,
    min_fraction: float = None,
    colors: dict[str, str] = None,
    title: str = None,
    selected_paraphrase: str = None,
    filename: str = None,
):
    if selected_categories is not None:
        df = df[df[category_column].isin(selected_categories)]

    if title is None and "question" in df.columns:
        questions = sorted(df["question"].unique())
        if selected_paraphrase is None:
            selected_paraphrase = questions[0]
        num_paraphrases = len(questions)
        if num_paraphrases == 1:
            title = selected_paraphrase
        else:
            title = selected_paraphrase + f"\nand {num_paraphrases - 1} other paraphrases"

    # Dispatch based on arguments and data
    stacked_bar_args = selected_answers is not None or min_fraction is not None or colors is not None

    if stacked_bar_args:
        # Stacked bar specific args provided
        non_null = df[answer_column].dropna()
        sample_value = non_null.iloc[0] if len(non_null) > 0 else None
        if isinstance(sample_value, dict):
            return probs_stacked_bar(
                df,
                probs_column=answer_column,
                category_column=category_column,
                selected_categories=selected_categories,
                selected_answers=selected_answers,
                min_fraction=min_fraction,
                colors=colors,
                title=title,
                filename=filename,
            )
        else:
            return free_form_stacked_bar(
                df,
                category_column=category_column,
                answer_column=answer_column,
                selected_categories=selected_categories,
                selected_answers=selected_answers,
                min_fraction=min_fraction,
                colors=colors,
                title=title,
                filename=filename,
            )

    # Check if data contains dicts with integer keys (rating probs)
    non_null = df[answer_column].dropna()
    sample_value = non_null.iloc[0] if len(non_null) > 0 else None
    if isinstance(sample_value, dict) and sample_value and all(isinstance(k, int) for k in sample_value.keys()):
        # Infer min_rating and max_rating from data if not provided
        if min_rating is None or max_rating is None:
            all_keys = set()
            for probs in df[answer_column].dropna():
                if isinstance(probs, dict):
                    all_keys.update(probs.keys())
            if all_keys:
                min_rating = min(all_keys)
                max_rating = max(all_keys)

        return rating_cumulative_plot(
            df,
            min_rating=min_rating,
            max_rating=max_rating,
            probs_column=answer_column,
            category_column=category_column,
            selected_categories=selected_categories,
            title=title,
            filename=filename,
        )
    elif isinstance(sample_value, dict):
        # Dict with non-integer keys (e.g., token probs)
        return probs_stacked_bar(
            df,
            probs_column=answer_column,
            category_column=category_column,
            selected_categories=selected_categories,
            title=title,
            filename=filename,
        )
    else:
        # Discrete values
        return free_form_stacked_bar(
            df,
            category_column=category_column,
            answer_column=answer_column,
            selected_categories=selected_categories,
            title=title,
            filename=filename,
        )


def rating_cumulative_plot(
    df: pd.DataFrame,
    min_rating: int,
    max_rating: int,
    probs_column: str = "probs",
    category_column: str = "group",
    selected_categories: list[str] = None,
    title: str = None,
    filename: str = None,
):
    categories = list(df[category_column].unique())
    if selected_categories is not None:
        categories = [c for c in selected_categories if c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = list(range(min_rating, max_rating + 1))

    for category in categories:
        category_df = df[df[category_column] == category]

        cumulative = {x: 0.0 for x in x_values}
        mean_sum = 0.0
        n_valid = 0

        for probs in category_df[probs_column]:
            if probs is None:
                continue

            for x in x_values:
                cumulative[x] += sum(p for rating, p in probs.items() if rating <= x)

            mean_sum += sum(rating * p for rating, p in probs.items())
            n_valid += 1

        if n_valid > 0:
            y_values = [cumulative[x] / n_valid for x in x_values]
            mean_value = mean_sum / n_valid
            label = f"{category} (mean: {mean_value:.1f})"
            ax.plot(x_values, y_values, label=label)

    ax.set_xlabel("Rating")
    ax.set_ylabel("Fraction with score ≤ X")
    ax.set_xlim(min_rating, max_rating)
    ax.set_ylim(0, 1)
    ax.legend()

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
    return fig


def probs_stacked_bar(
    df: pd.DataFrame,
    probs_column: str = "probs",
    category_column: str = "group",
    selected_categories: list[str] = None,
    selected_answers: list[str] = None,
    min_fraction: float = None,
    colors: dict[str, str] = None,
    title: str = None,
    filename: str = None,
):
    if min_fraction is not None and selected_answers is not None:
        raise ValueError("min_fraction and selected_answers cannot both be set")

    # Aggregate probs across rows for each category
    category_probs = {}
    for category in df[category_column].unique():
        cat_df = df[df[category_column] == category]
        combined = {}
        n_rows = 0
        for probs in cat_df[probs_column]:
            if probs is None:
                continue
            for answer, prob in probs.items():
                combined[answer] = combined.get(answer, 0) + prob
            n_rows += 1
        if n_rows > 0:
            category_probs[category] = {k: v / n_rows for k, v in combined.items()}

    if not category_probs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center", transform=ax.transAxes)
        if title is not None:
            ax.set_title(title)
        plt.show()
        return fig

    # Find answers meeting min_fraction threshold
    if min_fraction is not None:
        selected_answers_set = set()
        for probs in category_probs.values():
            for answer, prob in probs.items():
                if prob >= min_fraction:
                    selected_answers_set.add(answer)
        selected_answers = list(selected_answers_set)

    # Group non-selected answers into "[OTHER]"
    if selected_answers is not None:
        for category in category_probs:
            probs = category_probs[category]
            other_prob = sum(p for a, p in probs.items() if a not in selected_answers)
            category_probs[category] = {a: p for a, p in probs.items() if a in selected_answers}
            if other_prob > 0:
                category_probs[category]["[OTHER]"] = other_prob

    # Build percentages DataFrame
    all_answers = set()
    for probs in category_probs.values():
        all_answers.update(probs.keys())

    data = {cat: {a: probs.get(a, 0) * 100 for a in all_answers} for cat, probs in category_probs.items()}
    answer_percentages = pd.DataFrame(data).T

    # Color setup
    if colors is None:
        colors = {}
    if "[OTHER]" in all_answers and "[OTHER]" not in colors:
        colors["[OTHER]"] = "grey"

    color_palette = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "olive",
        "cyan",
        "magenta",
        "yellow",
        "navy",
        "lime",
        "maroon",
        "teal",
        "silver",
        "gold",
        "indigo",
        "coral",
        "crimson",
    ]

    # Order answers
    column_answers = list(answer_percentages.columns)
    if selected_answers is not None:
        ordered_answers = [a for a in selected_answers if a in column_answers]
        extras = sorted([a for a in column_answers if a not in selected_answers])
        ordered_answers += extras
    elif colors:
        ordered_answers = [a for a in colors.keys() if a in column_answers]
        extras = sorted([a for a in column_answers if a not in ordered_answers])
        ordered_answers += extras
    else:
        ordered_answers = sorted(column_answers)
    answer_percentages = answer_percentages.reindex(columns=ordered_answers)

    # Build colors list
    plot_colors = []
    color_index = 0
    for answer in ordered_answers:
        if answer in colors:
            plot_colors.append(colors[answer])
        elif answer == "[OTHER]":
            plot_colors.append("grey")
        else:
            plot_colors.append(color_palette[color_index % len(color_palette)])
            color_index += 1

    # Order categories
    if selected_categories is not None:
        ordered_categories = [c for c in selected_categories if c in answer_percentages.index]
        ordered_categories += [c for c in answer_percentages.index if c not in ordered_categories]
        answer_percentages = answer_percentages.reindex(ordered_categories)

    fig, ax = plt.subplots(figsize=(12, 8))
    answer_percentages.plot(kind="bar", stacked=True, ax=ax, color=plot_colors)

    plt.xlabel(category_column)
    plt.ylabel("Percentage")
    plt.legend(title="answer")
    plt.xticks(rotation=45, ha="right")

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
    return fig


def free_form_stacked_bar(
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
    probs_data = []
    for category in df[category_column].unique():
        cat_df = df[df[category_column] == category]
        counts = cat_df[answer_column].value_counts()
        probs = (counts / counts.sum()).to_dict()
        probs_data.append({category_column: category, "probs": probs})

    probs_df = pd.DataFrame(probs_data)

    return probs_stacked_bar(
        probs_df,
        probs_column="probs",
        category_column=category_column,
        selected_categories=selected_categories,
        selected_answers=selected_answers,
        min_fraction=min_fraction,
        colors=colors,
        title=title,
        filename=filename,
    )
