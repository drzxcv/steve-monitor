"""Evaluation: F1, Brier score, threshold choice, and bootstrap over world seeds."""
import numpy as np
from numpy.typing import ArrayLike


def confusion(y: ArrayLike, pred: ArrayLike) -> dict[str, int]:
    """Counts of tp, fp, fn, tn."""
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {
        "tp": int((y & pred).sum()), "fp": int((~y & pred).sum()),
        "fn": int((y & ~pred).sum()), "tn": int((~y & ~pred).sum()),
    }


def prf(y: ArrayLike, pred: ArrayLike) -> dict:
    """Precision, recall, F1 and the confusion counts."""
    c = confusion(y, pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **c}


def brier(y: ArrayLike, prob: ArrayLike) -> float:
    """Mean squared error of probabilities against 0/1 labels."""
    return float(np.mean((np.asarray(prob, float) - np.asarray(y, float)) ** 2))


def choose_threshold(y: ArrayLike, prob: ArrayLike) -> float:
    """Threshold with the best F1. Call this on validation data only."""
    grid = np.linspace(0.05, 0.95, 19)
    scores = [prf(y, np.asarray(prob) >= th)["f1"] for th in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y: ArrayLike, prob: ArrayLike, task: ArrayLike, threshold: float,
             task_names: list[str]) -> dict:
    """Per-task metrics plus the average of the per-task F1 scores."""
    y, prob, task = np.asarray(y), np.asarray(prob), np.asarray(task)
    out = {}
    for tid, name in enumerate(task_names):
        m = task == tid
        if not m.any():
            continue
        out[name] = {**prf(y[m], prob[m] >= threshold), "brier": brier(y[m], prob[m]),
                     "n": int(m.sum()), "positive_rate": float(y[m].mean())}
    out["macro_f1"] = float(np.mean([out[n]["f1"] for n in task_names if n in out]))
    return out


def bootstrap_macro_f1(y: ArrayLike, prob: ArrayLike, task: ArrayLike, seed: ArrayLike,
                       threshold: float, n_tasks: int, n_boot: int = 1000,
                       rng_seed: int = 0) -> dict[str, float]:
    """95% interval for macro F1, resampling world seeds with replacement.

    A world seed carries both of its task trials and all of their overlapping
    windows with it, so overlapping samples are never treated as independent.
    """
    y, task, seed = np.asarray(y).astype(bool), np.asarray(task), np.asarray(seed)
    pred = np.asarray(prob) >= threshold
    worlds = np.unique(seed)
    world_idx = np.searchsorted(worlds, seed)
    counts = np.zeros((len(worlds), n_tasks, 3))
    for k, m in enumerate((y & pred, ~y & pred, y & ~pred)):
        np.add.at(counts[:, :, k], (world_idx[m], task[m]), 1)
    rng = np.random.default_rng(rng_seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        tp, fp, fn = counts[rng.integers(0, len(worlds), len(worlds))].sum(axis=0).T
        denom = 2 * tp + fp + fn
        scores[b] = np.mean(np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"mean": float(scores.mean()), "lo": float(lo), "hi": float(hi)}
