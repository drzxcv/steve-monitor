"""Record STEVE-1 gameplay for a range of world seeds. Safe to stop and restart:
finished trials are skipped.

    python scripts/collect.py --split train --limit 5        # pilot
    python scripts/collect.py --split all                    # full run

WORKSTATION ONLY. Output: data/raw/<task>/seed_XXXXXX.npz.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from mcmon.config import REPO_ROOT, load_config
from mcmon.sim import run_trial_retry
from mcmon.splits import SPLITS, load_splits

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="all", choices=["all", *SPLITS])
parser.add_argument("--limit", type=int, default=None, help="max seeds per split")
parser.add_argument("--out", default="data/raw")
args = parser.parse_args()

cfg = load_config()
splits = load_splits(REPO_ROOT / cfg["splits"]["file"])
seeds = [s for name in SPLITS if args.split in ("all", name) for s in splits[name][:args.limit]]
sim = None
for seed in seeds:
    for task, tcfg in cfg["tasks"].items():
        path = Path(args.out) / task / f"seed_{seed:06d}.npz"
        if path.exists():
            continue
        start = time.time()
        frames, actions, counts, sim = run_trial_retry(cfg, task, seed, sim=sim)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, frames=frames, actions=actions, item_count=counts,
                            task=task, seed=seed, target=tcfg["target"])
        tmp.rename(path)   # a half-written file never looks finished
        done = int(np.argmax(counts >= tcfg["target"])) if (counts >= tcfg["target"]).any() else None
        print(json.dumps({"task": task, "seed": seed, "steps": len(counts), "final_count": int(counts[-1]),
                          "completed_at": done, "seconds": round(time.time() - start, 1)}), flush=True)
