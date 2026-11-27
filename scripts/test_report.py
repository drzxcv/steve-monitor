"""Summarise the final test runs: per-model scores and paired comparisons.

    python scripts/test_report.py

Reads runs/test_*/ (made by scripts/train.py --eval-test). Writes results/test_results.csv,
one row per run, and results/test_comparisons.csv, where each difference is bootstrapped
over test worlds with the same resampled worlds for both sides.
"""
import csv
import json

import numpy as np

from mcmon.config import REPO_ROOT, load_config

RUNS, RESULTS = REPO_ROOT / "runs", REPO_ROOT / "results"
N_TASKS = len(load_config()["task_names"])
SEEDS = (0, 1, 2)
GROUPS = {"xgb_h16": [f"test_xgb_h16_s{s}" for s in SEEDS], "xgb_h1": [f"test_xgb_h1_s{s}" for s in SEEDS],
          "cnn_h16": [f"test_cnn_h16_s{s}" for s in SEEDS], "cnn_h1": [f"test_cnn_h1_s{s}" for s in SEEDS],
          "time": ["test_time"]}
COMPARISONS = [("xgb_h16", "cnn_h16"), ("xgb_h16", "xgb_h1"), ("cnn_h16", "cnn_h1"),
               ("xgb_h16", "time"), ("cnn_h16", "time")]


def world_counts(run: str) -> np.ndarray:
    """[worlds, tasks, 3] counts of true positives, false positives and false negatives."""
    metrics = json.loads((RUNS / run / "metrics.json").read_text())
    with np.load(RUNS / run / "preds.npz") as z:
        y, pred = z["test_y"].astype(bool), z["test_prob"] >= metrics["threshold"]
        worlds, world_idx = np.unique(z["test_seed"], return_inverse=True)
        counts = np.zeros((len(worlds), N_TASKS, 3))
        for k, m in enumerate((y & pred, ~y & pred, y & ~pred)):
            np.add.at(counts[:, :, k], (world_idx[m], z["test_task"][m]), 1)
    return counts


def macro_f1(counts: np.ndarray) -> np.ndarray:
    """Macro-F1 from summed counts [..., tasks, 3]."""
    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0).mean(axis=-1)


def group_scores(group: str, picks: np.ndarray) -> np.ndarray:
    """Macro-F1 per bootstrap resample, averaged over the group's seeds."""
    return np.mean([macro_f1(world_counts(run)[picks].sum(axis=1)) for run in GROUPS[group]], axis=0)


rows = []
for group, runs in GROUPS.items():
    for run in runs:
        m = json.loads((RUNS / run / "metrics.json").read_text())
        row = {"group": group, "run": run, "threshold": m["threshold"],
               "val_macro_f1": round(m["val"]["macro_f1"], 4), "test_macro_f1": round(m["test"]["macro_f1"], 4),
               "ms_per_sample": round(m["test_ms_per_sample"], 5)}
        for task in ("logs", "dirt"):
            t = m["test"][task]
            row.update({f"{task}_{k}": round(t[k], 4) if isinstance(t[k], float) else t[k]
                        for k in ("f1", "precision", "recall", "brier", "tp", "fp", "fn", "tn")})
        rows.append(row)
with open(RESULTS / "test_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

n_worlds = len(world_counts("test_time"))
picks = np.random.default_rng(0).integers(0, n_worlds, (2000, n_worlds))
scores = {group: group_scores(group, picks) for group in GROUPS}
point = {group: float(np.mean([r["test_macro_f1"] for r in rows if r["group"] == group])) for group in GROUPS}
comparisons = []
for group in GROUPS:
    lo, hi = np.percentile(scores[group], [2.5, 97.5])
    comparisons.append({"quantity": group, "estimate": round(point[group], 4), "lo": round(lo, 4), "hi": round(hi, 4)})
for a, b in COMPARISONS:
    diff = scores[a] - scores[b]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    comparisons.append({"quantity": f"{a} - {b}", "estimate": round(point[a] - point[b], 4),
                        "lo": round(lo, 4), "hi": round(hi, 4)})
with open(RESULTS / "test_comparisons.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)
for c in comparisons:
    print(f"{c['quantity']:22s} {c['estimate']:+.3f}  [{c['lo']:+.3f}, {c['hi']:+.3f}]")
