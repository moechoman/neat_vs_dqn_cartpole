import os
import time
import argparse
from typing import Dict, Any

import yaml
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from utils import ensure_dir, set_global_seeds, save_json, make_env, now_str, evaluate_policy


class EvalLoggerCallback(BaseCallback):
    def __init__(self, cfg: Dict[str, Any], seed: int, run_dir: str):
        super().__init__()
        self.cfg = cfg
        self.seed = seed
        self.run_dir = run_dir

        self.eval_every = int(cfg["logging"]["eval_every_steps"])
        self.eval_episodes = int(cfg["budget"]["eval_episodes"])
        self.env_id = cfg["env_id"]

        self.records = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            # Evaluate deterministic policy
            def act_fn(obs):
                action, _ = self.model.predict(obs, deterministic=True,)
                return int(action)

            stats = evaluate_policy(
                env_id=self.env_id,
                act_fn=act_fn,
                seed=self.seed + 999,  # eval seed offset
                n_episodes=20,
            )

            row = {
                "env_steps": int(self.num_timesteps),
                "mean_reward_20ep": stats["mean_reward"],
                "std_reward_20ep": stats["std_reward"],
                "timestamp": time.time(),
            }
            self.records.append(row)

            pd.DataFrame(self.records).to_csv(
                os.path.join(self.run_dir, "dqn_progress.csv"),
                index=False
            )

        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(args.seed)
    set_global_seeds(seed)

    out_root = "results"
    ensure_dir(out_root)

    run_name = f"dqn_seed{seed}_{now_str()}"
    run_dir = os.path.join(out_root, "dqn", run_name)
    ensure_dir(run_dir)

    # Save config snapshot
    save_json(os.path.join(run_dir, "config_snapshot.json"), cfg)

    env_id = cfg["env_id"]
    max_steps = int(cfg["budget"]["max_env_steps"])

    env = make_env(env_id, seed=seed)

    dqn_cfg = cfg["dqn"]

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(dqn_cfg["learning_rate"]),
        gamma=float(dqn_cfg["gamma"]),
        buffer_size=int(dqn_cfg["buffer_size"]),
        batch_size=int(dqn_cfg["batch_size"]),
        learning_starts=int(dqn_cfg["learning_starts"]),
        target_update_interval=int(dqn_cfg["target_update_interval"]),
        train_freq=int(dqn_cfg["train_freq"]),
        gradient_steps=int(dqn_cfg["gradient_steps"]),
        exploration_fraction=float(dqn_cfg["exploration_fraction"]),
        exploration_final_eps=float(dqn_cfg["exploration_final_eps"]),
        verbose=0,
        seed=seed,
        device="auto",
    )

    callback = EvalLoggerCallback(cfg=cfg, seed=seed, run_dir=run_dir)

    t0 = time.time()
    model.learn(total_timesteps=max_steps, callback=callback, progress_bar=True)
    train_time = time.time() - t0

    model_path = os.path.join(run_dir, "dqn_model.zip")
    model.save(model_path)
    env.close()

    # Final evaluation (100 episodes)
    def act_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    final_stats = evaluate_policy(
        env_id=env_id,
        act_fn=act_fn,
        seed=seed + 999,
        n_episodes=int(cfg["budget"]["eval_episodes"]),
    )

    final_stats["train_time_sec"] = float(train_time)
    final_stats["max_env_steps"] = int(max_steps)

    save_json(os.path.join(run_dir, "final_eval.json"), final_stats)

    print(f"[DQN] Seed={seed} done.")
    print(f"Run dir: {run_dir}")
    print(f"Final mean reward: {final_stats['mean_reward']:.2f} ± {final_stats['std_reward']:.2f}")


if __name__ == "__main__":
    main()
