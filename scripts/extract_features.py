"""Turn recorded frames into feature vectors once, for reuse by every model.

    python scripts/extract_features.py --raw data/raw --out data/features
    python scripts/extract_features.py --hud hud_only --out data/features_hud_only   # leakage check

The hotbar and item pop-ups are blacked out first (`mcmon.hud`), because they show the
inventory the labels come from. Encoders are frozen and 512-d: an ImageNet ResNet-18
(the default) or MineCLIP's image encoder from the STEVE-1 checkpoint.
"""
import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from mcmon.data import Trial, save_trial
from mcmon.hud import MODES, apply_hud_mode


def build_encoder(name: str, device: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Frozen encoder as a function: uint8 frames [B, 3, H, W] -> features [B, 512]."""
    if name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        weights = ResNet18_Weights.IMAGENET1K_V1
        net, preprocess = resnet18(weights=weights), weights.transforms()
        net.fc = torch.nn.Identity()
        net.eval().to(device)
        return lambda frames: net(preprocess(frames).to(device))
    from mcmon.config import load_config
    from mcmon.sim import _load_policy
    mineclip = _load_policy(load_config()).mineclip

    def encode(frames: torch.Tensor) -> torch.Tensor:
        # Same resize STEVE-1 applies before MineCLIP; values stay in 0-255.
        x = torch.nn.functional.interpolate(frames.float().to(device), size=(160, 256),
                                            mode="bilinear", align_corners=False)
        return mineclip.forward_image_features(x)
    return encode


parser = argparse.ArgumentParser()
parser.add_argument("--raw", default="data/raw")
parser.add_argument("--out", default="data/features")
parser.add_argument("--batch", type=int, default=256)
parser.add_argument("--encoder", default="resnet18", choices=["resnet18", "mineclip"])
parser.add_argument("--hud", default="masked", choices=MODES, help="masked is the real experiment")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
encode = build_encoder(args.encoder, device)
for path in sorted(Path(args.raw).glob("*/seed_*.npz")):
    with np.load(path) as z:
        task, seed, target = str(z["task"]), int(z["seed"]), int(z["target"])
        if (Path(args.out) / task / path.name).exists():
            continue
        frames = torch.from_numpy(apply_hud_mode(z["frames"], args.hud)).permute(0, 3, 1, 2)
        feats = []
        with torch.no_grad():
            for batch in frames.split(args.batch):
                feats.append(encode(batch).float().cpu())
        save_trial(args.out, Trial(task, seed, target, torch.cat(feats).numpy(), z["actions"], z["item_count"]))
        print("features:", task, seed, flush=True)
