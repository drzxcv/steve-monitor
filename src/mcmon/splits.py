"""World-seed splits.

Seeds are assigned to train/val/test before any sample exists, and both tasks
for a seed live in the same split.
"""
import json
from pathlib import Path

import numpy as np

SPLITS = ("train", "val", "test")


def make_splits(n_seeds: int, base_seed: int, fractions: dict[str, float]) -> dict[str, list[int]]:
    """Draw distinct world seeds and divide them into train/val/test."""
    if abs(sum(fractions[s] for s in SPLITS) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    rng = np.random.default_rng(base_seed)
    seeds = rng.choice(np.arange(1, 1_000_000), size=n_seeds, replace=False)
    n_train = round(n_seeds * fractions["train"])
    n_val = round(n_seeds * fractions["val"])
    return {
        "train": sorted(int(s) for s in seeds[:n_train]),
        "val": sorted(int(s) for s in seeds[n_train:n_train + n_val]),
        "test": sorted(int(s) for s in seeds[n_train + n_val:]),
    }


def save_splits(splits: dict[str, list[int]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)
        f.write("\n")


def load_splits(path: str | Path) -> dict[str, list[int]]:
    """Read the splits file and check no seed is in two splits."""
    with open(path) as f:
        splits = json.load(f)
    check_disjoint(splits)
    return splits


def check_disjoint(splits: dict[str, list[int]]) -> None:
    """Raise ValueError if any seed appears in more than one split."""
    seen = {}
    for name in SPLITS:
        for seed in splits[name]:
            if seed in seen:
                raise ValueError(f"seed {seed} is in both {seen[seed]} and {name}")
            seen[seed] = name


def split_of(splits: dict[str, list[int]]) -> dict[int, str]:
    """Map each seed to its split name."""
    return {seed: name for name in SPLITS for seed in splits[name]}
