import pandas as pd
import matplotlib.pyplot as plt


def default_title(paraphrases: list[str] | None) -> str | None:
    """Generate default plot title from paraphrases."""
    if paraphrases is None:
        return None
    if len(paraphrases) == 1:
        return paraphrases[0]
    return paraphrases[0] + f"\nand {len(paraphrases) - 1} other paraphrases"


def rating_cumulative_plot(
    df: pd.DataFrame,
    min_rating: int,
    max_rating: int,
    probs_column: str = "probs",
    category_column: str = "group",
    model_groups: dict[str, list[str]] = None,
    show_mean: bool = True,
    title: str = None,
    filename: str = None,
):
    """Plot cumulative rating distribution by category.
    
    Shows fraction of responses with rating <= X for each X.
    Starts near 0 at min_rating, reaches 100% at max_rating.
    
    Args:
        df: DataFrame with probs_column containing normalized probability dicts
            mapping int ratings to probabilities (summing to 1), or None for invalid.
        min_rating: Minimum rating value.
        max_rating: Maximum rating value.
        probs_column: Column containing {rating: prob} dicts. Default: "probs"
        category_column: Column to group by. Default: "group"
        model_groups: Optional dict for ordering groups.
        show_mean: Whether to show mean in legend labels. Default: True
        title: Optional plot title.
        filename: Optional filename to save plot.
    """
    # Get unique categories in order
    categories = df[category_column].unique()
    if category_column == "group" and model_groups is not None:
        categories = [c for c in model_groups.keys() if c in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = list(range(min_rating, max_rating + 1))
    
    for category in categories:
        category_df = df[df[category_column] == category]
        
        # Accumulate normalized probabilities and means across all rows
        cumulative = {x: 0.0 for x in x_values}
        mean_sum = 0.0
        n_valid = 0
        
        for probs in category_df[probs_column]:
            if probs is None:
                continue
            
            # For each x, add P(score <= x) = sum of probs for ratings <= x
            for x in x_values:
                cumulative[x] += sum(p for rating, p in probs.items() if rating <= x)
            
            # Compute mean for this row
            mean_sum += sum(rating * p for rating, p in probs.items())
            n_valid += 1
        
        if n_valid > 0:
            y_values = [cumulative[x] / n_valid for x in x_values]
            mean_value = mean_sum / n_valid
            
            if show_mean:
                label = f"{category} (mean: {mean_value:.1f})"
            else:
                label = category
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


def probs_stacked_bar(
    df: pd.DataFrame,
    probs_column: str = "probs",
    category_column: str = "group",
    model_groups: dict[str, list[str]] = None,
    selected_answers: list[str] = None,
    min_fraction: float = None,
    colors: dict[str, str] = None,
    title: str = None,
    filename: str = None,
):
    """
    Plot a stacked bar chart from probability distributions.
    
    Args:
        df: DataFrame with one row per category, containing probs_column with
            {answer: probability} dicts.
        probs_column: Column containing probability dicts. Default: "probs"
        category_column: Column to group by (x-axis). Default: "group"
        model_groups: Optional dict for ordering groups.
        selected_answers: Optional list of answers to show. Others grouped as "[OTHER]".
        min_fraction: Optional minimum fraction threshold.
        colors: Optional dict mapping answer values to colors.
        title: Optional plot title.
        filename: Optional filename to save plot.
    """
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
        return

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
    
    data = {cat: {a: probs.get(a, 0) * 100 for a in all_answers} 
            for cat, probs in category_probs.items()}
    answer_percentages = pd.DataFrame(data).T

    # Color setup
    if colors is None:
        colors = {}
    if "[OTHER]" in all_answers and "[OTHER]" not in colors:
        colors["[OTHER]"] = "grey"

    color_palette = [
        "red", "blue", "green", "orange", "purple", "brown", "pink", "olive", 
        "cyan", "magenta", "yellow", "navy", "lime", "maroon", "teal", "silver",
        "gold", "indigo", "coral", "crimson"
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
    if category_column == "group" and model_groups is not None:
        ordered_groups = [g for g in model_groups.keys() if g in answer_percentages.index]
        ordered_groups += [g for g in answer_percentages.index if g not in ordered_groups]
        answer_percentages = answer_percentages.reindex(ordered_groups)

    fig, ax = plt.subplots(figsize=(12, 8))
    answer_percentages.plot(kind='bar', stacked=True, ax=ax, color=plot_colors)

    plt.xlabel(category_column)
    plt.ylabel('Percentage')
    plt.legend(title='answer')
    plt.xticks(rotation=45, ha='right')
    
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def free_form_stacked_bar(
    df: pd.DataFrame,
    category_column: str = "group",
    answer_column: str = "answer",
    model_groups: dict[str, list[str]] = None,
    selected_answers: list[str] = None,
    min_fraction: float = None,
    colors: dict[str, str] = None,
    title: str = None,
    filename: str = None,
):
    """
    Plot a stacked bar chart showing the distribution of answers by category.
    
    Transforms FreeForm data (multiple rows with single answers) into probability
    distributions and calls probs_stacked_bar.
    """
    # Transform to probs format: one row per category with {answer: prob} dict
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
        model_groups=model_groups,
        selected_answers=selected_answers,
        min_fraction=min_fraction,
        colors=colors,
        title=title,
        filename=filename,
    )

