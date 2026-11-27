# Results

## Test set (50 held-out worlds, run once)

`python scripts/test_report.py` writes `test_results.csv` (one row per run, with confusion
counts, Brier score and prediction time) and `test_comparisons.csv`. Macro-F1 is the mean of
three seeds. Intervals are 95% bootstrap intervals over test worlds; differences resample
the same worlds for both sides.

| Model | Test macro-F1 | 95% interval | Logs F1 | Dirt F1 | Validation |
| --- | --- | --- | --- | --- | --- |
| CNN, history 16 | 0.636 | 0.52 to 0.73 | 0.61 | 0.66 | 0.667 |
| XGBoost, history 16 | 0.624 | 0.53 to 0.70 | 0.56 | 0.69 | 0.704 |
| CNN, history 1 | 0.618 | 0.52 to 0.70 | 0.58 | 0.66 | 0.605 |
| XGBoost, history 1 | 0.607 | 0.51 to 0.69 | 0.53 | 0.68 | 0.662 |
| Time rule | 0.507 | 0.41 to 0.59 | 0.49 | 0.53 | 0.536 |

| Difference | Estimate | 95% interval |
| --- | --- | --- |
| XGBoost - CNN (history 16) | -0.012 | -0.085 to +0.067 |
| XGBoost: history 16 - history 1 | +0.017 | -0.007 to +0.042 |
| CNN: history 16 - history 1 | +0.018 | -0.030 to +0.067 |
| XGBoost - time rule | +0.117 | +0.052 to +0.180 |
| CNN - time rule | +0.128 | +0.054 to +0.193 |

Both learned monitors beat the time rule. The two model families cannot be separated, and
the gain from history is not distinguishable from zero. XGBoost fell further from validation
to test (0.70 to 0.62) than the CNN (0.67 to 0.64); validation data chose its configuration,
early-stopping point and threshold.

The sections below are validation data.

## Selected configurations (three seeds each)

`python scripts/search.py` runs the six configurations per model in `configs/search.yaml`
(`search_results.csv`). The best of each was retrained with seeds 0 to 2 at history 16 and at
history 1 (`validation_seeds.csv`). Macro-F1, mean and standard deviation over seeds:

| Model | History 16 | History 1 |
| --- | --- | --- |
| XGBoost, depth 4 | 0.704 (0.005) | 0.662 (0.004) |
| CNN, 128 filters, pooled readout | 0.667 (0.006) | 0.605 (0.019) |
| CNN, kernel 5, newest-step readout | 0.620 (0.007) | 0.613 (0.007) |
| Time rule | 0.536 | |

The CNN first read only the newest position of the window, which reaches back 4 frames with
kernel 3 and 7 with kernel 5, so it ignored most of the history. Its output layer now also
sees the max and mean over the whole window (`pool` in `src/mcmon/models/cnn.py`). All six
pooled configurations beat all six original ones, and history now helps the CNN (+0.06)
as it helps XGBoost (+0.04). The original runs are kept as `cnn_last_step`.

## Encoder check (three seeds, history 16)

Both selected configurations were retrained on MineCLIP features (`encoder_check.csv`).
ResNet-18 is kept: it is better for XGBoost and no worse for the CNN.

| Model | ResNet-18 | MineCLIP |
| --- | --- | --- |
| XGBoost, depth 4 | 0.704 (0.005) | 0.675 (0.009) |
| CNN, 128 filters, pooled readout | 0.667 (0.006) | 0.655 (0.021) |

## Pilot (default model settings, one seed)

The pilot XGBoost runs used 256 histogram bins; the default is now 64, which needs a third
of the memory and scores 0.69 where the table below shows 0.68.

Regenerate with `python scripts/leakage_check.py` then `python scripts/pilot_report.py`.
Every number is in `pilot_results.csv`, one row per `scripts/train.py` run with its input settings.

| Input | Model | Macro-F1 | Logs F1 | Dirt F1 |
| --- | --- | --- | --- | --- |
| Raw hotbar pixels, one frame (leak check) | small pixel CNN | 0.97 | 0.99 | 0.94 |
| Unmasked frames, ResNet-18 | XGBoost | 0.62 | 0.55 | 0.69 |
| Masked frames, ResNet-18, 16 frames over 480 steps + action totals (config default) | XGBoost | 0.68 | 0.63 | 0.73 |
| Same | CNN | 0.60 | 0.52 | 0.67 |
| Masked frames, ResNet-18, last 16 steps | XGBoost | 0.59 | 0.50 | 0.68 |
| Masked frames, MineCLIP, last 16 steps | XGBoost | 0.59 | 0.51 | 0.66 |
| No image (actions, task, time) | XGBoost | 0.55 | 0.50 | 0.60 |
| Time rule | | 0.54 | 0.47 | 0.60 |

What the pilot shows:

1. **The hotbar leaks the label.** A small CNN on the bottom strip of a single raw frame
   scores 0.97. The HUD (hotbar, health, chat text, pick-up pop-ups) is blacked out before
   feature extraction (`src/mcmon/hud.py`, `figures/hud_mask.png`).
2. **Encoder choice does not matter here.** ResNet-18 (ImageNet) and MineCLIP's image encoder
   give the same score on masked frames. ResNet-18 is kept: it is independent of STEVE-1 and
   needs no simulator install.
3. **Memory matters more than the encoder.** Completion depends on everything collected since
   the start, but 16 consecutive steps cover 1% of a trial. Spacing the 16 frames 32 steps apart
   and adding running action counts lifts XGBoost from 0.59 to 0.68.

Figures: `figures/input_comparison.png`, `figures/hud_mask.png`.
