# Setup

## Modelling (any machine)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Without a GPU, install CPU PyTorch first to save a large download:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## GPU workstation: simulator and STEVE-1

Linux (or WSL) with an NVIDIA GPU.

```bash
conda env create -f environment.yml      # Python 3.10, JDK 8, MineStudio 1.1.6, this package
conda activate mcmon
sudo apt install -y xvfb mesa-utils libegl1-mesa libgl1-mesa-dev libglu1-mesa-dev
export MINESTUDIO_DIR=$HOME/MineStudio   # also in ~/.bashrc; the default location under /tmp is wiped on restart
python scripts/smoke_test_env.py --task logs --steps 300
```

The first run downloads the Minecraft engine (about 450 MB) and the STEVE-1 weights. The
smoke test passes when frames come back and the inventory count starts at 0. Many `WARN`
and `ERROR` lines from Minecraft (Narrator, Realms, OptiFine textures) are harmless.

## Pilot

`scripts/pilot_sweep.py` compares prompts and starting tools on training worlds without
saving frames. The collected data uses the `prompt`, `init_commands` and `target` in
`configs/default.yaml`.

Results on the first 15 training worlds, random spawns, 1,200 steps, MineStudio 1.1.6.
Final count per world:

| Task | Setting | Final counts | Reached 4 |
| --- | --- | --- | --- |
| logs | bare hands, any of 3 prompts | all 0 | 0/15 |
| logs | axe, "chop tree", guidance 4, goal re-sampled every step | 5,0,0,2,0,0,0,0,4,0,1,0,2,4,7 | 4/15 |
| logs | **axe, "chop tree", guidance 6, goal fixed per trial (current)** | 6,0,4,3,0,0,3,7,1,0,2,3,1,3,3 | 3/15 |
| dirt | bare hands, any of 3 prompts | max 8 | 3/15 |
| dirt | shovel, "collect dirt", guidance 4, re-sampled | 8,0,5,2,4,9,0,0,1,0,1,0,4,2,0 | 5/15 |
| dirt | **shovel, "collect dirt", guidance 6, fixed (current)** | 4,0,6,3,6,2,4,0,1,0,1,9,1,9,0 | 6/15 |

Other settings tried on 5 to 10 worlds did no better: "chop a tree" and "break a log" with an
axe, "get dirt" and "dig as far as possible" with a shovel, and forest spawns. No dirt trial
reached the original target of 16.

The full collection (500 trials) took about 7 hours and 12 GB, plus 1 GB of features.
Completed trials per split (train / val / test): logs 59/150, 22/50, 21/50; dirt 72/150,
27/50, 21/50.
