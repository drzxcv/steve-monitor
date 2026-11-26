"""Collect the pilot runs into results/ for the report and slides.

    python scripts/pilot_report.py

Reads runs/pilot_*/metrics.json and results/leakage_check.json. Writes
results/pilot_results.csv and two figures under results/figures/.
"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mcmon.config import REPO_ROOT
from mcmon.hud import apply_hud_mode

RESULTS = REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

rows = []
for path in sorted((REPO_ROOT / "runs").glob("pilot_*/metrics.json")):
    r = json.loads(path.read_text())
    rows.append({"run": path.parent.name, "model": r["model"], "data": r.get("data", ""),
                 "no_image": r.get("no_image", False), "history": r["inputs"]["history"],
                 "frame_gap": r["inputs"]["frame_gap"],
                 "running_totals": r["inputs"].get("running_totals", False),
                 "threshold": r["threshold"], "val_macro_f1": round(r["val"]["macro_f1"], 4),
                 "val_f1_logs": round(r["val"]["logs"]["f1"], 4), "val_f1_dirt": round(r["val"]["dirt"]["f1"], 4)})
leak = json.loads((RESULTS / "leakage_check.json").read_text())
rows.append({"run": "leakage_check", "model": "pixel cnn", "data": "raw HUD strip", "history": 1,
             "threshold": leak["threshold"], "val_macro_f1": round(leak["val"]["macro_f1"], 4),
             "val_f1_logs": round(leak["val"]["logs"]["f1"], 4), "val_f1_dirt": round(leak["val"]["dirt"]["f1"], 4)})
with open(RESULTS / "pilot_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
score = {r["run"]: r["val_macro_f1"] for r in rows}

# Figure 1: what each input is worth (XGBoost unless noted).
HIDDEN, VISIBLE, INK, MUTED = "#257de0", "#f47230", "#191919", "#666666"
bars = [
    ("Raw hotbar pixels, one frame (leak check)", "leakage_check", VISIBLE),
    ("Unmasked frames, ResNet-18", "pilot_unmasked_xgb", VISIBLE),
    ("Masked frames, wide window + action totals", "pilot_gap32_totals_xgb", HIDDEN),
    ("Masked frames, MineCLIP", "pilot_mineclip_xgb", HIDDEN),
    ("Masked frames, ResNet-18", "pilot_masked_xgb", HIDDEN),
    ("No image: actions + time", "pilot_noimage_xgb", HIDDEN),
    ("Time rule", "pilot_time", HIDDEN),
]
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
y = np.arange(len(bars))[::-1]
values = [score[run] for _, run, _ in bars]
ax.barh(y, values, height=0.62, color=[c for *_, c in bars])
for yi, v in zip(y, values):
    ax.text(v + 0.012, yi, f"{v:.2f}", va="center", color=INK, fontsize=10)
ax.set_yticks(y, [label for label, *_ in bars], color=INK, fontsize=10)
ax.set_xlim(0, 1.08)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="x", colors=MUTED, labelsize=9)
ax.tick_params(length=0)
ax.set_xlabel("Validation macro-F1", color=MUTED, fontsize=9)
ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
for side in ax.spines.values():
    side.set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=VISIBLE), plt.Rectangle((0, 0), 1, 1, color=HIDDEN)],
          labels=["Hotbar visible", "Hotbar hidden"], loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
ax.set_title("The hotbar gives the label away; without it the task is hard", loc="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "input_comparison.png")

# Figure 2: one frame as recorded, as the monitors see it, and the part removed.
raw = sorted((REPO_ROOT / "data" / "raw" / "logs").glob("seed_*.npz"))
if raw:
    with np.load(raw[0]) as z:
        t = int(np.argmax(z["item_count"] >= int(z["target"]))) + 5
        frame = z["frames"][t:t + 1]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.3), dpi=200)
    for ax, mode, title in zip(axes, ("none", "masked", "hud_only"),
                               ("Recorded frame", "Monitor input (HUD masked)", "Removed region")):
        ax.imshow(apply_hud_mode(frame, mode)[0], interpolation="nearest")
        ax.set_title(title, fontsize=10, color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "hud_mask.png")
print("wrote", RESULTS)
