import os
import json
import time
import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import numpy as np
import gymnasium as gym


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_env(env_id: str, seed: int, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def now_str() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def episode_rollout(env: gym.Env, act_fn, max_steps: int = 500) -> Tuple[float, int]:
    """
    Run exactly one episode and return total reward + used steps.
    """
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0

    for _ in range(max_steps):
        action = act_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_r += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return total_r, steps


def evaluate_policy(env_id: str, act_fn, seed: int, n_episodes: int = 100) -> Dict[str, Any]:
    """
    Deterministic evaluation over n_episodes.
    """
    env = make_env(env_id, seed=seed)
    rewards = []
    steps_total = 0

    for ep in range(n_episodes):
        r, steps = episode_rollout(env, act_fn)
        rewards.append(r)
        steps_total += steps

    env.close()

    rewards = np.array(rewards, dtype=float)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "episodes": int(n_episodes),
        "eval_env_steps": int(steps_total),
    }
