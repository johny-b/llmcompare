"""Utility functions for llmcompare."""

import json
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, data: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file.
    
    Each dictionary is written as a JSON object on a separate line.
    
    Args:
        path: Path to the output JSONL file
        data: List of dictionaries to write
        
    Example:
        >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        >>> write_jsonl("people.jsonl", data)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries.
    
    Each line is parsed as a JSON object.
    
    Args:
        path: Path to the input JSONL file
        
    Returns:
        List of dictionaries, one per line in the file
        
    Example:
        >>> data = read_jsonl("people.jsonl")
        >>> print(data)
        [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
    """
    path = Path(path)
    data = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    
    return data

def get_error_bars(fraction_list, rng=None, alpha=0.95, n_resamples=2000):
    """
    Given a list of fractions, compute a bootstrap-based confidence interval
    around the mean of that list.

    Returns:
      (center, lower_err, upper_err)
    where:
      - center = mean of fraction_list
      - lower_err = center - lower_CI
      - upper_err = upper_CI - center

    So if you want to pass these to plt.errorbar as yerr:
      yerr = [[lower_err], [upper_err]]
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)
    fractions = np.array(fraction_list, dtype=float)

    # Edge cases
    if len(fractions) == 0:
        return (0.0, 0.0, 0.0)
    if len(fractions) == 1:
        return (fractions[0], 0.0, 0.0)

    boot_means = []
    for _ in range(n_resamples):
        sample = rng.choice(fractions, size=len(fractions), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)

    lower_bound = np.percentile(boot_means, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(boot_means, (1 - (1 - alpha) / 2) * 100)
    center = np.mean(fractions)

    lower_err = center - lower_bound
    upper_err = upper_bound - center

    return (center, lower_err, upper_err)