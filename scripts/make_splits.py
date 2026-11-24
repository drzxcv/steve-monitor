"""Write the world-seed splits file. Run once; the file is committed and then frozen."""
import argparse
from pathlib import Path

from mcmon.config import REPO_ROOT, load_config
from mcmon.splits import make_splits, save_splits

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="overwrite an existing splits file")
args = parser.parse_args()

cfg = load_config()["splits"]
path = REPO_ROOT / cfg["file"]
if Path(path).exists() and not args.force:
    raise SystemExit(f"{path} already exists. Changing it invalidates collected data; use --force if you mean it.")
splits = make_splits(cfg["n_seeds"], cfg["base_seed"], cfg["fractions"])
save_splits(splits, path)
print({k: len(v) for k, v in splits.items()}, "->", path)
