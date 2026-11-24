"""Train one monitor and report metrics.

    python scripts/train.py --model xgb --data data/features --name xgb_h16
    python scripts/train.py --model cnn --data data/features --name cnn_h1 --history 1

Validation metrics are always reported. Test metrics are only computed with
--eval-test, which is for the final selected configurations.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from mcmon.config import REPO_ROOT, load_config
from mcmon.data import build_samples, load_trials
from mcmon.metrics import bootstrap_macro_f1, choose_threshold, evaluate
from mcmon.models.cnn import CNNMonitor
from mcmon.models.time_rule import TimeRule
from mcmon.models.trees import TreeMonitor
from mcmon.splits import load_splits

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["time", "xgb", "cnn"], required=True)
parser.add_argument("--data", required=True, help="folder of feature trials")
parser.add_argument("--name", required=True, help="results go to runs/<name>/")
parser.add_argument("--history", type=int, default=None, help="override inputs.history")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--params", default="{}", help="JSON of model settings, e.g. '{\"max_depth\": 4}'")
parser.add_argument("--eval-test", action="store_true")
args = parser.parse_args()

cfg = load_config()
splits = load_splits(REPO_ROOT / cfg["splits"]["file"])
inputs = dict(cfg["inputs"])
if args.history is not None:
    inputs["history"] = args.history
names, max_steps = cfg["task_names"], cfg["collection"]["max_steps"]


def samples_for(split):
    trials = load_trials(args.data, splits[split])
    if not trials:
        raise SystemExit(f"no {split} trials found under {args.data}")
    return build_samples(trials, names, max_steps, **inputs)


train, val = samples_for("train"), samples_for("val")
params = json.loads(args.params)
if args.model == "time":
    model = TimeRule().fit(train, len(names))
elif args.model == "xgb":
    model = TreeMonitor(seed=args.seed, **params).fit(train, val)
else:
    model = CNNMonitor(seed=args.seed, epochs=args.epochs, **params).fit(train, val)

val_prob = model.predict_proba(val)
threshold = 0.5 if args.model == "time" else choose_threshold(val.y, val_prob)
result = {
    "model": args.model, "seed": args.seed, "inputs": inputs, "params": params,
    "threshold": threshold, "n_train": len(train.y),
    "val": evaluate(val.y, val_prob, val.task, threshold, names),
}
out = REPO_ROOT / "runs" / args.name
out.mkdir(parents=True, exist_ok=True)
preds = {"val_prob": val_prob, "val_y": val.y, "val_task": val.task, "val_seed": val.seed, "val_step": val.step}

if args.eval_test:
    test = samples_for("test")
    start = time.perf_counter()
    test_prob = model.predict_proba(test)
    result["test_ms_per_sample"] = 1000 * (time.perf_counter() - start) / len(test.y)
    result["test"] = evaluate(test.y, test_prob, test.task, threshold, names)
    result["test"]["macro_f1_bootstrap"] = bootstrap_macro_f1(
        test.y, test_prob, test.task, test.seed, threshold, len(names))
    preds.update(test_prob=test_prob, test_y=test.y, test_task=test.task, test_seed=test.seed, test_step=test.step)

np.savez_compressed(out / "preds.npz", **preds)
(out / "metrics.json").write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps({k: result[k] for k in ("model", "threshold", "val") if k in result}, indent=1))
if "test" in result:
    print("test:", json.dumps(result["test"], indent=1))
print("saved to", out)
