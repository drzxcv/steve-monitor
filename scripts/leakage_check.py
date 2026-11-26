"""Label-leakage check: predict completion from the raw HUD pixels of a single frame.

    python scripts/leakage_check.py

A high score means the label can be read off the hotbar, so the HUD has to be masked
before feature extraction. Uses data/raw; writes results/leakage_check.json.
"""
import argparse
import json

import numpy as np
import torch
from torch import nn

from mcmon.config import REPO_ROOT, load_config
from mcmon.hud import HUD_REGIONS
from mcmon.metrics import choose_threshold, evaluate
from mcmon.splits import load_splits

parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--stride", type=int, default=4)
args = parser.parse_args()

cfg = load_config()
splits = load_splits(REPO_ROOT / cfg["splits"]["file"])
names = cfg["task_names"]
top = int(HUD_REGIONS[0][0] * cfg["collection"]["obs_size"])   # the bottom strip only


def load(split: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Bottom-strip crops, labels and task ids for one split."""
    crops, labels, tasks = [], [], []
    for tid, task in enumerate(names):
        for seed in splits[split]:
            with np.load(f"{args.raw}/{task}/seed_{seed:06d}.npz") as z:
                crops.append(z["frames"][::args.stride, top:].copy())   # copy, or every full frame array stays in memory
                labels.append(z["item_count"][::args.stride] >= cfg["tasks"][task]["target"])
                tasks.append(np.full(len(labels[-1]), tid))
    x = torch.from_numpy(np.concatenate(crops)).permute(0, 3, 1, 2)
    return x, np.concatenate(labels).astype(np.int8), np.concatenate(tasks)


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d((2, 16)), nn.Flatten(),
    nn.Linear(64 * 32, 128), nn.ReLU(), nn.Linear(128, 1)).to(device)
opt = torch.optim.AdamW(net.parameters(), 1e-3)
(x, y, _), (xv, yv, tv) = load("train"), load("val")
yt = torch.from_numpy(y).float()
for _ in range(args.epochs):
    for idx in torch.randperm(len(x)).split(256):
        logits = net(x[idx].to(device).float() / 255).squeeze(1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, yt[idx].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
net.eval()
with torch.no_grad():
    prob = torch.cat([torch.sigmoid(net(b.to(device).float() / 255)).squeeze(1).cpu() for b in xv.split(2048)]).numpy()
threshold = choose_threshold(yv, prob)
result = {"input": "raw HUD strip, one frame", "epochs": args.epochs, "threshold": threshold,
          "val": evaluate(yv, prob, tv, threshold, names)}
out = REPO_ROOT / "results"
out.mkdir(exist_ok=True)
(out / "leakage_check.json").write_text(json.dumps(result, indent=1) + "\n")
print(json.dumps(result["val"], indent=1))
