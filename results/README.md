# Pilot results (validation only, default model settings, one seed)

Regenerate with `python scripts/leakage_check.py` then `python scripts/pilot_report.py`.
Every number is in `pilot_results.csv`, one row per `scripts/train.py` run with its input settings.
No test data has been used.

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
