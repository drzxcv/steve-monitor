"""Turn recorded frames into feature vectors once, for reuse by every model.

    python scripts/extract_features.py --raw data/raw --out data/features

The encoder is frozen. The default is an ImageNet ResNet-18 (512-d) because it
installs anywhere; swapping in MineCLIP or the VPT image encoder is a change to
`build_encoder` only. Pick one in the pilot and do not change it afterwards.
"""
import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from mcmon.data import Trial, save_trial


def build_encoder(device: str) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Frozen image encoder and its preprocessing transform."""
    from torchvision.models import ResNet18_Weights, resnet18
    weights = ResNet18_Weights.IMAGENET1K_V1
    net = resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    return net.eval().to(device), weights.transforms()


parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw")
parser.add_argument("--out", default="data/features")
parser.add_argument("--batch", type=int, default=256)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder, preprocess = build_encoder(device)
for path in sorted(Path(args.raw).glob("*/seed_*.npz")):
    with np.load(path) as z:
        task, seed, target = str(z["task"]), int(z["seed"]), int(z["target"])
        if (Path(args.out) / task / path.name).exists():
            continue
        frames = torch.from_numpy(z["frames"]).permute(0, 3, 1, 2)
        feats = []
        with torch.no_grad():
            for batch in frames.split(args.batch):
                feats.append(encoder(preprocess(batch).to(device)).cpu())
        save_trial(args.out, Trial(task, seed, target, torch.cat(feats).numpy(), z["actions"], z["item_count"]))
        print("features:", task, seed, flush=True)
