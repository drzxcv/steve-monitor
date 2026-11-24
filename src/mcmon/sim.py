"""Run one STEVE-1 trial in the MineStudio simulator and record it.

WORKSTATION ONLY: imports minestudio, which the modelling code never needs.
Tested with MineStudio 1.1.6.
"""
from typing import Any

import numpy as np

from .labels import count_item

# Saved action vector: these 20 buttons, then camera pitch and yaw / 180.
BUTTONS = ["attack", "back", "forward", "jump", "left", "right", "sneak", "sprint", "use", "drop",
           "inventory"] + [f"hotbar.{i}" for i in range(1, 10)]
ACTION_DIM = len(BUTTONS) + 2
_policy = None


def _load_policy(cfg: dict) -> Any:
    """Load STEVE-1 once and reuse it."""
    global _policy
    if _policy is None:
        import torch
        from minestudio.models import SteveOnePolicy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _policy = SteveOnePolicy.from_pretrained(cfg["collection"]["policy"]).to(device)
        _policy.eval()
    return _policy


def _prepare_goal(policy: Any, prompt: str, cond_scale: float) -> dict:
    """Sample STEVE-1's goal embedding from the text prior once, for a whole trial.

    MineStudio's get_action would re-sample it from the prior at every step.
    """
    import torch
    with torch.inference_mode():
        return policy.prepare_condition({"cond_scale": cond_scale, "text": prompt})


def _act(policy: Any, image: np.ndarray, goal: dict, memory: Any) -> tuple[dict, Any]:
    """One policy step. Same as SteveOnePolicy.get_action(input_shape="*"), but with a fixed goal."""
    import torch
    from minestudio.models.base_policy import recursive_tensor_op
    with torch.inference_mode():
        img = torch.from_numpy(np.ascontiguousarray(image)).to(policy.device)[None, None]
        state = None if memory is None else recursive_tensor_op(lambda x: x.unsqueeze(0), memory)
        latents, state_out = policy.forward({"image": img, "condition": goal}, state)
        action = policy.pi_head.sample(latents["pi_logits"], False)
        return ({k: v[0][0] for k, v in action.items()},
                recursive_tensor_op(lambda x: x[0], state_out))


def _action_vector(env_action: dict) -> np.ndarray:
    """Flatten an env action dict into the fixed vector saved to disk."""
    vec = np.zeros(ACTION_DIM, np.float32)
    for i, name in enumerate(BUTTONS):
        vec[i] = float(np.asarray(env_action.get(name, 0)).reshape(-1)[0])
    vec[-2:] = np.asarray(env_action["camera"], np.float32).reshape(-1)[:2] / 180.0
    return vec


def run_trial(cfg: dict, task: str, seed: int, max_steps: int | None = None, sim: Any = None,
              verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Returns (frames [T,H,W,3] uint8, actions [T,A], item_count [T], sim).

    Row t holds the frame seen at step t, the inventory count at that moment,
    and the action the agent then took. Recording continues after the target
    is reached. Pass the returned sim back in to avoid restarting Minecraft.
    """
    from minestudio.simulator import MinecraftSim

    tcfg, ccfg = cfg["tasks"][task], cfg["collection"]
    max_steps = max_steps or ccfg["max_steps"]
    policy = _load_policy(cfg)
    if sim is None:
        size = ccfg["obs_size"]
        biome = ccfg.get("spawn_biome")
        sim = MinecraftSim(obs_size=(size, size), seed=seed,
                           **({"preferred_spawn_biome": biome} if biome else {}))
    sim.env.seed(seed)
    obs, info = sim.reset()
    # Same pattern as MineStudio's CommandsCallback.
    for command in tcfg.get("init_commands", []):
        raw_obs, _, _, raw_info = sim.env.execute_cmd(command)
        obs, info = sim._wrap_obs_info(raw_obs, raw_info)
    goal = _prepare_goal(policy, tcfg["prompt"], ccfg["cond_scale"])

    frames, actions, counts, memory = [], [], [], None
    for t in range(max_steps):
        count = count_item(info["inventory"], tcfg["item_pattern"])
        action, memory = _act(policy, obs["image"], goal, memory)
        env_action = sim.agent_action_to_env_action(dict(action))
        frames.append(obs["image"])
        actions.append(_action_vector(env_action))
        counts.append(count)
        if verbose and t % 50 == 0:
            print(f"step {t}: {task} count = {count}", flush=True)
        obs, _, terminated, truncated, info = sim.step(action)
        if terminated or truncated:
            break
    return np.stack(frames).astype(np.uint8), np.stack(actions), np.asarray(counts, np.int32), sim


def run_trial_retry(cfg: dict, task: str, seed: int, sim: Any = None, retries: int = 2,
                    **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """run_trial, restarting Minecraft and retrying if the simulator connection fails."""
    for attempt in range(retries + 1):
        try:
            return run_trial(cfg, task, seed, sim=sim, **kwargs)
        except Exception as err:      # Malmo failures surface as many exception types
            if attempt == retries:
                raise
            print(f"simulator error on {task} seed {seed} ({type(err).__name__}: {err}); restarting", flush=True)
            try:
                if sim is not None:
                    sim.close()
            except Exception:
                pass
            sim = None
