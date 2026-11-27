"""Run the hyperparameter search in configs/search.yaml and tabulate validation scores.

    python scripts/search.py --data data/features

One scripts/train.py run per configuration, saved as runs/search_<model>_<name>/.
Finished runs are skipped. Writes results/search_results.csv.
"""
import argparse
import csv
import json
import subprocess
import sys

import yaml

from mcmon.config import REPO_ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/features")
parser.add_argument("--grid", default="configs/search.yaml")
args = parser.parse_args()

grid = yaml.safe_load((REPO_ROOT / args.grid).read_text())
rows = []
for group, configs in grid.items():
    model = group.split("_")[0]   # "cnn_last_step" trains a cnn
    for config in configs:
        run = f"search_{group}_{config['name']}"
        metrics = REPO_ROOT / "runs" / run / "metrics.json"
        if not metrics.exists():
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--model", model,
                            "--data", args.data, "--name", run, "--params", json.dumps(config["params"])],
                           check=True, stdout=subprocess.DEVNULL)
        val = json.loads(metrics.read_text())["val"]
        rows.append({"model": group, "config": config["name"], "params": json.dumps(config["params"]),
                     "val_macro_f1": round(val["macro_f1"], 4),
                     **{f"val_f1_{t}": round(v["f1"], 4) for t, v in val.items() if t != "macro_f1"},
                     **{f"val_brier_{t}": round(v["brier"], 4) for t, v in val.items() if t != "macro_f1"}})
        print(rows[-1], flush=True)

out = REPO_ROOT / "results" / "search_results.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("wrote", out)
