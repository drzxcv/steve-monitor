"""Pilot: try prompts and starting commands on a few training worlds, without saving frames.

    python scripts/pilot_sweep.py --task logs --prompts "chop tree" "break a log" \
        --commands "/give @s minecraft:wooden_axe" --n-seeds 5

Prints one JSON line per trial and a summary per prompt. Uses training seeds only,
so choosing settings here never looks at validation or test worlds.
"""
import argparse
import json
import time

import numpy as np

from mcmon.config import REPO_ROOT, load_config
from mcmon.sim import run_trial_retry
from mcmon.splits import load_splits

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--prompts", nargs="+", required=True)
parser.add_argument("--commands", nargs="*", default=[], help="init commands, replacing the config's")
parser.add_argument("--biome", default=None, help="preferred spawn biome (default: config)")
parser.add_argument("--n-seeds", type=int, default=5)
parser.add_argument("--offset", type=int, default=0, help="skip the first N training seeds")
parser.add_argument("--steps", type=int, default=None)
args = parser.parse_args()

cfg = load_config()
tcfg = cfg["tasks"][args.task]
tcfg["init_commands"] = args.commands
if args.biome:
    cfg["collection"]["spawn_biome"] = args.biome
seeds = load_splits(REPO_ROOT / cfg["splits"]["file"])["train"][args.offset:args.offset + args.n_seeds]
sim = None
for prompt in args.prompts:
    tcfg["prompt"] = prompt
    finals, done_at = [], []
    for seed in seeds:
        start = time.time()
        _, _, counts, sim = run_trial_retry(cfg, args.task, seed, sim=sim, max_steps=args.steps)
        hit = np.flatnonzero(counts >= tcfg["target"])
        finals.append(int(counts[-1]))
        done_at.append(int(hit[0]) if len(hit) else None)
        print(json.dumps({"task": args.task, "prompt": prompt, "commands": args.commands, "seed": seed,
                          "final_count": finals[-1], "completed_at": done_at[-1],
                          "seconds": round(time.time() - start, 1)}), flush=True)
    print(json.dumps({"SUMMARY": True, "task": args.task, "prompt": prompt, "commands": args.commands,
                      "biome": cfg["collection"].get("spawn_biome"), "final_counts": finals,
                      "completed": sum(d is not None for d in done_at), "n": len(seeds)}), flush=True)
