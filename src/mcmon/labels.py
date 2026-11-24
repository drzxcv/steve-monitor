"""Completion labels from inventory counts.

Inventory is used for labels only. It must never reach a model as an input.
"""
import re

import numpy as np
from numpy.typing import ArrayLike


def count_item(inventory: dict, item_pattern: str) -> int:
    """Total quantity of items whose name matches the regex.

    `inventory` is the simulator's `info["inventory"]`: a dict of
    slot -> {"type": str, "quantity": int}.
    """
    pattern = re.compile(item_pattern)
    total = 0
    for slot in inventory.values():
        if pattern.search(str(slot["type"])):
            total += int(slot["quantity"])
    return total


def labels_from_counts(item_count: ArrayLike, target: int) -> np.ndarray:
    """1 where the current inventory meets the target, else 0."""
    return (np.asarray(item_count) >= target).astype(np.int8)
