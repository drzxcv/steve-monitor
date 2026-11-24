"""Generate fake trials for every seed and task, in the same layout as real features."""
import argparse

from mcmon.config import REPO_ROOT, load_config
from mcmon.data import save_trial
from mcmon.splits import SPLITS, load_splits
from mcmon.synthetic import make_trial

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/synthetic")
parser.add_argument("--max-seeds", type=int, default=None, help="per split, for a quick run")
args = parser.parse_args()

cfg = load_config()
splits = load_splits(REPO_ROOT / cfg["splits"]["file"])
n = 0
for split in SPLITS:
    for seed in splits[split][:args.max_seeds]:
        for task, tcfg in cfg["tasks"].items():
            save_trial(args.out, make_trial(task, seed, tcfg["target"], cfg["collection"]["max_steps"]))
            n += 1
print(f"wrote {n} synthetic trials to {args.out}")
