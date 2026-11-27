"""Trial files (format in README.md) and model inputs."""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import labels_from_counts


@dataclass
class Trial:
    """One recorded episode of one task in one world."""

    task: str
    seed: int
    target: int
    features: np.ndarray    # [T, D]
    actions: np.ndarray     # [T, A], chosen after seeing step t
    item_count: np.ndarray  # [T], labels only

    @property
    def labels(self) -> np.ndarray:
        return labels_from_counts(self.item_count, self.target)

    def __len__(self) -> int:
        return len(self.features)


def trial_path(root: str | Path, task: str, seed: int) -> Path:
    """Where a trial lives: <root>/<task>/seed_XXXXXX.npz."""
    return Path(root) / task / f"seed_{seed:06d}.npz"


def save_trial(root: str | Path, trial: Trial) -> Path:
    """Write a trial and return its path."""
    path = trial_path(root, trial.task, trial.seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=trial.features.astype(np.float32),
        actions=trial.actions.astype(np.float32),
        item_count=trial.item_count.astype(np.int32),
        task=trial.task, seed=trial.seed, target=trial.target,
    )
    return path


def load_trial(path: str | Path) -> Trial:
    """Read one trial file and check its arrays line up."""
    with np.load(path) as z:
        trial = Trial(
            task=str(z["task"]), seed=int(z["seed"]), target=int(z["target"]),
            features=z["features"], actions=z["actions"], item_count=z["item_count"],
        )
    n = len(trial.features)
    if not (len(trial.actions) == n and len(trial.item_count) == n):
        raise ValueError(f"{path}: features, actions and item_count differ in length")
    return trial


def load_trials(root: str | Path, seeds: Iterable[int] | None = None) -> list[Trial]:
    """Load every trial under root, optionally keeping only the given world seeds."""
    keep = None if seeds is None else set(seeds)
    trials = [load_trial(p) for p in sorted(Path(root).glob("*/seed_*.npz"))]
    return [t for t in trials if keep is None or t.seed in keep]


@dataclass
class Samples:
    """Model-ready windows plus the bookkeeping needed for evaluation."""

    x: np.ndarray        # [N, history, C], oldest step first
    y: np.ndarray        # [N]
    seed: np.ndarray     # [N] world seed
    task: np.ndarray     # [N] task id
    step: np.ndarray     # [N] step of the newest observation

    def flat(self) -> np.ndarray:
        """Windows joined into one vector per sample, for the trees."""
        return self.x.reshape(len(self.x), -1)


def build_samples(trials: Iterable[Trial], task_names: Sequence[str], max_steps: int, history: int, stride: int = 1,
                  frame_gap: int = 1, running_totals: bool = False) -> Samples:
    """Turn trials into fixed-length windows.

    Each step's channels are [features, action, task one-hot, elapsed fraction],
    plus the sum of past actions over max_steps when running_totals is set.
    The window for step t holds steps t, t-gap, ..., t-(history-1)*gap. Steps
    before the start of the trial repeat step 0, so no future data is ever used.

    The action stored at step t was chosen after seeing step t, so the newest
    position in a window uses the action from step t-1 (zeros at t=0).
    """
    trials = list(trials)
    offsets = np.arange(history - 1, -1, -1) * frame_gap
    n_samples = sum(len(range(0, len(trial), stride)) for trial in trials)
    x = None   # filled in place: a list of windows would need twice the memory
    ys, seeds, tasks, steps, row = [], [], [], [], 0
    for trial in trials:
        n = len(trial)
        tid = task_names.index(trial.task)
        onehot = np.zeros((n, len(task_names)), np.float32)
        onehot[:, tid] = 1.0
        elapsed = (np.arange(n, dtype=np.float32) / max_steps)[:, None]
        prev_actions = np.zeros_like(trial.actions, dtype=np.float32)
        prev_actions[1:] = trial.actions[:-1]
        per_step = np.concatenate(
            [trial.features.astype(np.float32), prev_actions, onehot, elapsed], axis=1)
        if running_totals:
            per_step = np.concatenate([per_step, np.cumsum(prev_actions, axis=0) / max_steps], axis=1)

        t = np.arange(0, n, stride)
        idx = np.clip(t[:, None] - offsets[None, :], 0, None)
        if x is None:
            x = np.empty((n_samples, history, per_step.shape[1]), np.float32)
        x[row:row + len(t)] = per_step[idx]
        row += len(t)
        ys.append(trial.labels[t])
        seeds.append(np.full(len(t), trial.seed))
        tasks.append(np.full(len(t), tid))
        steps.append(t)
    return Samples(
        x=x, y=np.concatenate(ys).astype(np.int8),
        seed=np.concatenate(seeds), task=np.concatenate(tasks), step=np.concatenate(steps),
    )
