"""Load the shared YAML config."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the config and add `task_names` (alphabetical, which fixes the task ids)."""
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["task_names"] = sorted(cfg["tasks"])
    return cfg


def task_id(cfg: dict, task: str) -> int:
    """Integer id of a task name."""
    return cfg["task_names"].index(task)
