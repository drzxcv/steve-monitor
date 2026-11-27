# Recognizing Task Completion in Minecraft

Boosted trees vs. a temporal CNN as task-completion monitors for a frozen STEVE-1 agent.
Track 4 project by Deautaun Ross and Chase Calodney.

A monitor looks at gameplay features and actions and predicts whether the task
(collect 4 logs, or collect 4 dirt) is finished. Inventory counts give the labels and
are never a model input, and the on-screen hotbar is masked out of every frame.

## Quick start

```bash
git clone https://github.com/drzxcv/steve-monitor.git && cd steve-monitor
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -e ".[dev]"
pytest -q
make demo                                              # fake data -> all three models -> metrics
```

This covers the modelling code and runs on any machine. Recording gameplay needs a GPU
workstation with the simulator: see [docs/SETUP.md](docs/SETUP.md).

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

`scripts/make_synthetic.py` writes fake feature files in the same format, which the tests
and `make demo` use.

## Layout

| Path | What it is |
| --- | --- |
| `configs/default.yaml` | Shared settings: tasks, prompts, targets, input window, split sizes |
| `configs/splits.json` | The 250 world seeds and their train/val/test split. Frozen. |
| `src/mcmon/` | Library: labels, data and windows, metrics, simulator, models |
| `scripts/` | One entry point per pipeline step, plus the search, checks and reports |
| `tests/` | Leakage, split, metric and training checks |
| `docs/SETUP.md` | Workstation setup and the collection pilot |
| `results/` | Result tables and figures kept in git |

## Safeguards

- **Fixed splits.** World seeds are assigned to train, validation and test before any data is
  collected, and both trials from a world stay in the same split. A test checks that
  `configs/splits.json` matches the config.
- **No label leakage.** Inventory counts are used only for labels. Model inputs are built in
  one place (`build_samples`), and a test checks they do not change when the inventory does.
  `extract_features.py` blacks out the hotbar and item pop-ups, which show the inventory on
  screen; `scripts/leakage_check.py` measures how much they would give away.
- **Test data used once.** Configurations, early stopping and thresholds are chosen on
  validation data. `train.py` computes test metrics only with `--eval-test`.
- **Data outside git.** Recorded gameplay (12 GB), features (1 GB) and run outputs are not
  tracked. STEVE-1 samples its actions, so a new collection would not reproduce the same trials.

GitHub Actions runs the tests on every pull request.

## Results

Tables and figures are in [results/](results/README.md): the test-set comparison of the two
monitors and the time rule, the hyperparameter search, the encoder comparison, and the
hotbar leakage check.
