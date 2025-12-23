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
        
        # Accumulate normalized probabilities across all rows
        cumulative = {x: 0.0 for x in x_values}
        n_valid = 0
        
        for probs in category_df[probs_column]:
            if probs is None:
                continue
            
            # For each x, add P(score <= x) = sum of probs for ratings <= x
            for x in x_values:
                cumulative[x] += sum(p for rating, p in probs.items() if rating <= x)
            n_valid += 1
        
        if n_valid > 0:
            y_values = [cumulative[x] / n_valid for x in x_values]
            ax.plot(x_values, y_values, label=category)
    
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
    
    Args:
        df: DataFrame containing the data to plot. Must contain columns specified by
            category_column and answer_column.
        category_column: Column name to use for grouping (x-axis). Default: "group"
        answer_column: Column name containing the answers to plot (stacked bars). Default: "answer"
        model_groups: Optional dict mapping group names to model lists. Used for ordering
            groups when category_column is "group". If None, groups are ordered by appearance.
        selected_answers: Optional list of specific answers to show. Others will be grouped
            as "[OTHER ANSWER]". Cannot be used with min_fraction.
        min_fraction: Optional minimum fraction threshold. Answers meeting this threshold
            in any category will be shown. Cannot be used with selected_answers.
        colors: Optional dict mapping answer values to color names/hex codes.
        title: Optional plot title. If None, no title is shown.
        filename: Optional filename to save the plot. If None, plot is displayed.
    """

    if min_fraction is not None and selected_answers is not None:
        raise ValueError("min_fraction and selected_answers cannot both be set")

    if min_fraction is not None:
        # For each category, find answers that meet the minimum fraction threshold
        selected_answers_set = set()
        
        # Group by category and calculate answer frequencies for each category
        for category in df[category_column].unique():
            category_df = df[df[category_column] == category]
            answer_counts = category_df[answer_column].value_counts()
            total_count = len(category_df)
            
            # Find answers that meet the minimum fraction threshold for this category
            for answer, count in answer_counts.items():
                fraction = count / total_count
                if fraction >= min_fraction:
                    selected_answers_set.add(answer)
        
        selected_answers = list(selected_answers_set)

    if selected_answers is not None:
        df = df.copy()
        df[answer_column] = df[answer_column].apply(lambda x: x if x in selected_answers else "[OTHER ANSWER]")

    if colors is None:
        colors = {}
    if "[OTHER ANSWER]" in df[answer_column].values:
        if "[OTHER ANSWER]" not in colors:
            colors["[OTHER ANSWER]"] = "grey"

    # Count occurrences of each answer for each category
    answer_counts = df.groupby([category_column, answer_column]).size().unstack(fill_value=0)

    # Convert to percentages
    answer_percentages = answer_counts.div(answer_counts.sum(axis=1), axis=0) * 100

    # ---- Legend & plotting order logic ----

    # Color palette for fallback
    color_palette = [
        "red", "blue", "green", "orange", "purple", "brown", "pink", "olive", 
        "cyan", "magenta", "yellow", "navy", "lime", "maroon", "teal", "silver",
        "gold", "indigo", "coral", "crimson"
    ]

    # Prepare legend/answer order
    column_answers = list(answer_percentages.columns)
    if selected_answers is not None:
        # Legend order: exactly selected_answers, then any remaining in alpha order
        ordered_answers = [a for a in selected_answers if a in column_answers]
        extras = sorted([a for a in column_answers if a not in selected_answers])
        ordered_answers += extras
    elif colors:
        color_keys = list(colors.keys())
        ordered_answers = [a for a in color_keys if a in column_answers]
        extras = sorted([a for a in column_answers if a not in ordered_answers])
        ordered_answers += extras
    else:
        ordered_answers = sorted(column_answers)
    # Reindex columns to the desired order
    answer_percentages = answer_percentages.reindex(columns=ordered_answers)

    # Build plot_colors according to legend order
    plot_colors = []
    color_index = 0
    for answer in ordered_answers:
        if answer in colors:
            plot_colors.append(colors[answer])
        elif answer == "[OTHER ANSWER]":
            plot_colors.append("grey")
        else:
            plot_colors.append(color_palette[color_index % len(color_palette)])
            color_index += 1

    # Determine ordering of groups if category_column is "group"
    if category_column == "group" and model_groups is not None:
        # Ensure plotting order matches the order of model_groups.keys()
        ordered_groups = [g for g in model_groups.keys() if g in answer_percentages.index]
        # Add any missing groups at the end (unlikely unless inconsistent data)
        ordered_groups += [g for g in answer_percentages.index if g not in ordered_groups]
        answer_percentages = answer_percentages.reindex(ordered_groups)

    fig, ax = plt.subplots(figsize=(12, 8))
    answer_percentages.plot(kind='bar', stacked=True, ax=ax, color=plot_colors)

    plt.xlabel(category_column)
    plt.ylabel('Percentage')
    plt.legend(title=answer_column)

    # Sort x-axis ticks by group if category_column is "model"
    if category_column == "model" and "group" in df.columns:
        tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
        model_to_group = df.groupby('model')['group'].first().to_dict()
        sorted_labels = sorted(tick_labels, key=lambda model: model_to_group.get(model, ''))
        answer_percentages_sorted = answer_percentages.reindex(sorted_labels)
        ax.clear()
        answer_percentages_sorted.plot(kind='bar', stacked=True, ax=ax, color=plot_colors)
        plt.xlabel(category_column)
        plt.ylabel('Percentage')
        plt.legend(title=answer_column)

    plt.xticks(rotation=45, ha='right')
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()

