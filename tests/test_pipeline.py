import numpy as np
import pytest

from mcmon.config import REPO_ROOT, load_config
from mcmon.data import Trial, build_samples, load_trial, load_trials, save_trial
from mcmon.hud import apply_hud_mode
from mcmon.labels import count_item, labels_from_counts
from mcmon.metrics import bootstrap_macro_f1, brier, choose_threshold, evaluate, prf
from mcmon.models.cnn import CNNMonitor
from mcmon.models.time_rule import TimeRule
from mcmon.models.trees import TreeMonitor
from mcmon.splits import SPLITS, check_disjoint, load_splits, make_splits
from mcmon.synthetic import make_trial

NAMES = ["dirt", "logs"]


def test_committed_splits_match_config():
    cfg = load_config()
    splits = load_splits(REPO_ROOT / cfg["splits"]["file"])
    assert splits == make_splits(cfg["splits"]["n_seeds"], cfg["splits"]["base_seed"], cfg["splits"]["fractions"])
    assert [len(splits[s]) for s in SPLITS] == [150, 50, 50]
    assert len({s for name in SPLITS for s in splits[name]}) == 250


def test_overlapping_splits_are_rejected():
    with pytest.raises(ValueError):
        check_disjoint({"train": [1, 2], "val": [2], "test": [3]})


def test_count_item_and_labels():
    inv = {0: {"type": "oak_log", "quantity": 3}, 1: {"type": "birch_log", "quantity": 2},
           2: {"type": "dirt", "quantity": 7}, 3: {"type": "coarse_dirt", "quantity": 9},
           4: {"type": "none", "quantity": 0}}
    cfg = load_config()["tasks"]
    assert count_item(inv, cfg["logs"]["item_pattern"]) == 5
    assert count_item(inv, cfg["dirt"]["item_pattern"]) == 7
    assert labels_from_counts([0, 3, 4, 5, 2], 4).tolist() == [0, 0, 1, 1, 0]


def test_trial_round_trip(tmp_path):
    trial = make_trial("logs", 7, target=4, max_steps=50)
    loaded = load_trial(save_trial(tmp_path, trial))
    assert loaded.task == "logs" and loaded.seed == 7 and loaded.target == 4
    np.testing.assert_array_equal(loaded.item_count, trial.item_count)
    assert [t.seed for t in load_trials(tmp_path, seeds=[7])] == [7]
    assert load_trials(tmp_path, seeds=[8]) == []


def _marker_trial(n=20):
    # Feature value equals the step index, so windows are easy to read.
    feats = np.arange(n, dtype=np.float32)[:, None]
    acts = 100 + np.arange(n, dtype=np.float32)[:, None]
    return Trial("logs", 1, 4, feats, acts, np.arange(n))


def test_windows_use_only_the_past():
    s = build_samples([_marker_trial()], NAMES, max_steps=20, history=4, stride=5)
    assert s.x.shape == (4, 4, 1 + 1 + 2 + 1)
    assert s.step.tolist() == [0, 5, 10, 15]
    assert s.x[2, :, 0].tolist() == [7, 8, 9, 10]          # oldest first, newest is step t
    assert s.x[0, :, 0].tolist() == [0, 0, 0, 0]           # padded with step 0, never the future
    assert s.x[2, -1, 1] == 109                            # action is the one taken before step t
    assert s.x[0, -1, 1] == 0
    assert s.x[2, -1, 2:4].tolist() == [0, 1]              # task one-hot for "logs"
    assert s.x[2, -1, 4] == pytest.approx(10 / 20)         # elapsed fraction
    assert s.y.tolist() == [0, 1, 1, 1]


def test_frame_gap_and_flat():
    s = build_samples([_marker_trial()], NAMES, max_steps=20, history=3, stride=10, frame_gap=4)
    assert s.x[1, :, 0].tolist() == [2, 6, 10]
    assert s.flat().shape == (2, 15)


def test_inputs_do_not_change_when_inventory_changes():
    a = _marker_trial()
    b = Trial(a.task, a.seed, a.target, a.features, a.actions, a.item_count[::-1].copy())
    xa = build_samples([a], NAMES, 20, history=4, running_totals=True).x
    xb = build_samples([b], NAMES, 20, history=4, running_totals=True).x
    np.testing.assert_array_equal(xa, xb)


def test_metrics():
    y = np.array([1, 1, 0, 0, 1])
    m = prf(y, np.array([1, 0, 1, 0, 1]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["f1"] == pytest.approx(2 / 3)
    assert prf([0, 0], [0, 0])["f1"] == 0.0
    assert brier([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    assert choose_threshold(y, np.array([0.9, 0.8, 0.2, 0.1, 0.7])) <= 0.7


def test_bootstrap_matches_point_estimate_scale():
    rng = np.random.default_rng(0)
    seed = np.repeat(np.arange(40), 50)
    task = np.tile(np.repeat([0, 1], 25), 40)
    y = rng.integers(0, 2, len(seed))
    prob = np.clip(y * 0.3 + rng.random(len(seed)) * 0.7, 0, 1)   # overlapping, so F1 < 1
    point = evaluate(y, prob, task, 0.5, NAMES)["macro_f1"]
    ci = bootstrap_macro_f1(y, prob, task, seed, 0.5, 2, n_boot=300)
    assert ci["lo"] < point < ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.2


@pytest.fixture(scope="module")
def synthetic():
    def make(seeds):
        trials = [make_trial(t, s, target, max_steps=300) for s in seeds for t, target in (("logs", 4), ("dirt", 16))]
        return build_samples(trials, NAMES, max_steps=300, history=8, stride=4)
    return make(range(30)), make(range(100, 110))


@pytest.mark.parametrize("build", [
    lambda: TimeRule(), lambda: TreeMonitor(n_estimators=40), lambda: CNNMonitor(epochs=3, device="cpu")])
def test_models_train_and_beat_chance(synthetic, build):
    train, val = synthetic
    model = build()
    model = model.fit(train, len(NAMES)) if isinstance(model, TimeRule) else model.fit(train, val)
    prob = model.predict_proba(val)
    assert prob.shape == val.y.shape and prob.min() >= 0 and prob.max() <= 1
    threshold = 0.5 if isinstance(model, TimeRule) else choose_threshold(val.y, prob)
    always_yes = prf(val.y, np.ones_like(val.y))["f1"]
    assert evaluate(val.y, prob, val.task, threshold, NAMES)["macro_f1"] > always_yes - 0.05


def test_running_totals_count_past_actions_only():
    s = build_samples([_marker_trial()], NAMES, max_steps=20, history=1, stride=5, running_totals=True)
    assert s.x.shape == (4, 1, 1 + 1 + 2 + 1 + 1)
    assert s.x[1, 0, -1] == pytest.approx(sum(100 + i for i in range(5)) / 20)   # actions 0..4 at step 5


def test_hud_mask_splits_the_frame():
    frames = np.full((2, 128, 128, 3), 255, dtype=np.uint8)
    masked, hud = apply_hud_mode(frames, "masked"), apply_hud_mode(frames, "hud_only")
    assert (masked[:, 110:, :] == 0).all() and (masked[:, :30, 100:] == 0).all()
    assert (masked[:, 50, 50] == 255).all()
    assert ((masked > 0) ^ (hud > 0)).all()
