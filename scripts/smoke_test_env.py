"""Simulator check on the GPU workstation: does the simulator run, does STEVE-1 act,
and do inventory counts come back every step?

    python scripts/smoke_test_env.py --task logs --steps 300
    python scripts/smoke_test_env.py --task dirt --prompt "get dirt"

WORKSTATION ONLY. Needs MineStudio, JDK 8 and a display (Xvfb or VirtualGL);
see docs/SETUP.md.
"""
import argparse

from mcmon.config import load_config
from mcmon.sim import run_trial

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="logs")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--prompt", default=None, help="override the task prompt from the config")
args = parser.parse_args()

cfg = load_config()
if args.prompt:
    cfg["tasks"][args.task]["prompt"] = args.prompt
frames, actions, counts, _ = run_trial(cfg, args.task, args.seed, max_steps=args.steps, verbose=True)
print(f"frames {frames.shape} {frames.dtype}, actions {actions.shape}")
print(f"start count {counts[0]}, final count {counts[-1]}, target {cfg['tasks'][args.task]['target']}")
assert counts[0] == 0, "inventory should start with none of the target item"
print("OK: simulator, policy and inventory readout all work.")
