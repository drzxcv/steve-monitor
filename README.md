# Recognizing Task Completion in Minecraft

Boosted trees vs. a temporal CNN as task-completion monitors for a frozen STEVE-1 agent.
Track 4 project by Deautaun Ross and Chase Calodney.

A monitor looks at recent gameplay features and actions and predicts whether the task
(collect 4 logs, or collect 4 dirt) is finished. Inventory counts give the labels and
are never a model input.

## Start here

```bash
git clone https://github.com/drzxcv/steve-monitor.git && cd steve-monitor
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -e ".[dev]"
pytest -q
make demo                                              # fake data -> all three models -> metrics
```

That is all you need for modelling work on any machine. Recording gameplay needs the GPU
workstation: see [docs/SETUP.md](docs/SETUP.md).

## Pipeline

```
scripts/collect.py            STEVE-1 plays; saves frames, actions, inventory counts   (workstation)
        |  data/raw/<task>/seed_XXXXXX.npz
scripts/extract_features.py   frozen visual encoder, run once                          (workstation)
        |  data/features/<task>/seed_XXXXXX.npz
scripts/train.py              time rule, XGBoost or CNN; threshold picked on validation (anywhere)
        |  runs/<name>/metrics.json, preds.npz
```

Each `.npz` is one trial (one task in one world). Row `t` of every array is the same moment:

| Key | Shape | Meaning |
| --- | --- | --- |
| `frames` (raw) / `features` | `[T, 128, 128, 3]` uint8 / `[T, D]` float32 | What the agent saw at step `t` |
| `actions` | `[T, 22]` float32 | Action chosen after step `t`: 20 buttons, then camera / 180 |
| `item_count` | `[T]` int32 | Target items held at step `t`. Labels only. |
| `task`, `seed`, `target` | scalars | Label is `item_count >= target` |

`scripts/make_synthetic.py` writes fake feature files, so modelling work does not wait for
collection.

## Layout

| Path | What it is |
| --- | --- |
| `configs/default.yaml` | Shared settings: tasks, prompts, targets, history length, split sizes |
| `configs/splits.json` | The 250 world seeds and their train/val/test split. Frozen. |
| `src/mcmon/` | Library: labels, data and windows, metrics, simulator, models |
| `scripts/` | One entry point per pipeline step |
| `tests/` | Leakage, split, metric and training checks |
| `docs/SETUP.md` | Workstation setup and pilot results |

## Rules that protect the results

1. **Never edit `configs/splits.json`** after collection starts. A test checks it matches the config.
2. **Inventory is for labels only.** Inputs are built only in `build_samples`, and a test checks they ignore inventory.
3. **Tune on validation.** `train.py` reports test metrics only with `--eval-test`; use it for final configurations only.
4. **Data and results stay out of git.** Share `data/features/` (about 1 GB) through a shared drive and keep raw data on the workstation. STEVE-1 samples its actions, so a re-run is not identical: back up the collected data.

## Working together

Work on short branches and review each other's pull requests. GitHub Actions runs the tests
on every pull request. Put changes to `configs/default.yaml` in their own pull request.

## Status

Tested: splits, labels, windowing, metrics with world-seed bootstrap, time rule, XGBoost, CNN,
synthetic data, training, and the simulator side (MineStudio 1.1.6), including a pilot.

Still open:
- Visual encoder: ResNet-18 is a placeholder in `scripts/extract_features.py`. Fix it before the full extraction.
- Hyperparameter search (up to six configurations per model), three seeds each, history vs. latest frame, error analysis.
