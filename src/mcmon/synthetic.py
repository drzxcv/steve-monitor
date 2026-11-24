"""Fake trials in the real file format, for building the pipeline before data exists.

The signal is deliberately simple: a few feature dimensions drift as items are
picked up, some trials never complete, and noise is high enough that a single
frame is less reliable than a short history. Results on this data mean nothing
about Minecraft.
"""
import numpy as np

from .data import Trial


def make_trial(task: str, seed: int, target: int, max_steps: int = 1200, feat_dim: int = 32,
               action_dim: int = 22, rng: np.random.Generator | None = None) -> Trial:
    """One fake trial; deterministic for a given task and seed."""
    rng = rng or np.random.default_rng([seed, sum(map(ord, task))])
    pickup_rate = rng.uniform(0.3, 2.5) * target / max_steps
    count = np.cumsum(rng.random(max_steps) < pickup_rate).astype(np.int32)
    count[0] = 0
    progress = np.minimum(count / target, 1.5)
    features = rng.normal(0, 1.0, (max_steps, feat_dim)).astype(np.float32)
    features[:, 0] += progress
    features[:, 1] += 0.5 * (count >= target)
    actions = (rng.random((max_steps, action_dim)) < 0.2).astype(np.float32)
    return Trial(task, seed, target, features, actions, count)
