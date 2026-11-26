# Recognizing Task Completion in Minecraft

Boosted trees vs. a temporal CNN as task-completion monitors for a frozen STEVE-1 agent.
Track 4 project by Deautaun Ross and Chase Calodney.

A monitor looks at gameplay features and actions and predicts whether the task
(collect 4 logs, or collect 4 dirt) is finished. Inventory counts give the labels and
are never a model input, and the on-screen hotbar is masked out of every frame.

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
scripts/extract_features.py   masks the HUD, then a frozen ResNet-18; run once         (workstation)
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
| `configs/default.yaml` | Shared settings: tasks, prompts, targets, input window, split sizes |
| `configs/splits.json` | The 250 world seeds and their train/val/test split. Frozen. |
| `src/mcmon/` | Library: labels, data and windows, metrics, simulator, models |
| `scripts/` | One entry point per pipeline step |
| `tests/` | Leakage, split, metric and training checks |
| `docs/SETUP.md` | Workstation setup and the collection pilot |
| `results/` | Result tables and figures kept in git |

## Rules that protect the results

1. **Never edit `configs/splits.json`** after collection starts. A test checks it matches the config.
2. **Inventory is for labels only.** Inputs are built only in `build_samples`, and a test checks they ignore inventory. `extract_features.py` blacks out the hotbar and item pop-ups, which show the inventory on screen.
3. **Tune on validation.** `train.py` reports test metrics only with `--eval-test`; use it for final configurations only.
4. **Data and run outputs stay out of git.** Share `data/features/` (about 1 GB) through a shared drive and keep raw data on the workstation. STEVE-1 samples its actions, so a re-run is not identical: back up the collected data.

## Working together

Work on short branches and review each other's pull requests. GitHub Actions runs the tests
on every pull request. Put changes to `configs/default.yaml` in their own pull request.

## Results

Pilot results on validation data are in [results/](results/README.md): the hotbar leakage
check, the encoder comparison, and the effect of a longer input window.
