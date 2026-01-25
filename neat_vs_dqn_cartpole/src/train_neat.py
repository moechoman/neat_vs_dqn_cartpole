import os
import time
import argparse
import pickle
from typing import Dict, Any, Tuple

import yaml
import numpy as np
import pandas as pd
import gymnasium as gym
import neat

from utils import ensure_dir, set_global_seeds, save_json, make_env, now_str, episode_rollout, evaluate_policy


class BudgetTracker:
    def __init__(self, max_env_steps: int):
        self.max_env_steps = int(max_env_steps)
        self.used_steps = 0

    def add(self, n: int) -> None:
        self.used_steps += int(n)

    def exhausted(self) -> bool:
        return self.used_steps >= self.max_env_steps


def make_neat_act_fn(net) -> callable:
    def act(obs):
        out = net.activate(obs)
        # output is 2 numbers, choose argmax
        return int(np.argmax(out))
    return act


def eval_genome(
    genome,
    config,
    env_id: str,
    base_seed: int,
    tracker: BudgetTracker,
) -> Tuple[float, int]:
    """
    Evaluate a single genome on ONE episode (Option 1).
    Returns fitness (reward) and env_steps used.
    """
    if tracker.exhausted():
        return 0.0, 0

    env = make_env(env_id, seed=base_seed)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    act_fn = make_neat_act_fn(net)

    reward, steps = episode_rollout(env, act_fn)
    env.close()

    tracker.add(steps)
    return float(reward), int(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--neat_cfg", type=str, default="configs/neat.cfg")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(args.seed)
    set_global_seeds(seed)

    out_root = "results"
    ensure_dir(out_root)

    run_name = f"neat_seed{seed}_{now_str()}"
    run_dir = os.path.join(out_root, "neat", run_name)
    ensure_dir(run_dir)

    save_json(os.path.join(run_dir, "config_snapshot.json"), cfg)

    env_id = cfg["env_id"]
    max_steps = int(cfg["budget"]["max_env_steps"])

    neat_cfg_path = args.neat_cfg
    neat_config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        neat_cfg_path,
    )

    # Override pop size if experiment.yaml specifies it
    neat_config.pop_size = int(cfg["neat"]["pop_size"])

    pop = neat.Population(neat_config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    tracker = BudgetTracker(max_env_steps=max_steps)
    records = []

    t0 = time.time()

    generation = 0
    winner = None

    while generation < int(cfg["neat"]["max_generations"]):
        generation += 1

        if tracker.exhausted():
            break

        # NEAT expects a fitness function of form (genomes, config)
        def fitness_fn(genomes, config):
            for gid, genome in genomes:
                # deterministic but diversified seed per genome + generation
                ep_seed = seed * 10_000 + generation * 100 + (gid % 100)
                fit, used = eval_genome(
                    genome=genome,
                    config=config,
                    env_id=env_id,
                    base_seed=ep_seed,
                    tracker=tracker,
                )
                genome.fitness = fit

        pop.run(fitness_fn, 1)  # run exactly ONE generation

        best_genome = stats.best_genome()
        best_fit = float(best_genome.fitness if best_genome else 0.0)

        row = {
            "generation": generation,
            "best_fitness": best_fit,
            "env_steps_used": tracker.used_steps,
            "timestamp": time.time(),
        }
        records.append(row)

        pd.DataFrame(records).to_csv(
            os.path.join(run_dir, "neat_progress.csv"),
            index=False
        )

        winner = best_genome

        # quick print
        if generation % 5 == 0 or generation == 1:
            print(f"[NEAT] gen={generation}, best={best_fit:.1f}, steps={tracker.used_steps}/{max_steps}")

    train_time = time.time() - t0

    # Save winner genome
    winner_path = os.path.join(run_dir, "neat_winner.pkl")
    with open(winner_path, "wb") as f:
        pickle.dump(winner, f)

    # Final evaluation (100 episodes)
    if winner is None:
        final_stats = {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "episodes": int(cfg["budget"]["eval_episodes"]),
        }
    else:
        net = neat.nn.FeedForwardNetwork.create(winner, neat_config)
        act_fn = make_neat_act_fn(net)

        final_stats = evaluate_policy(
            env_id=env_id,
            act_fn=act_fn,
            seed=seed + 999,
            n_episodes=int(cfg["budget"]["eval_episodes"]),
        )

    final_stats["train_time_sec"] = float(train_time)
    final_stats["max_env_steps"] = int(max_steps)
    final_stats["env_steps_used"] = int(tracker.used_steps)
    final_stats["generations_ran"] = int(generation)

    save_json(os.path.join(run_dir, "final_eval.json"), final_stats)

    print(f"[NEAT] Seed={seed} done.")
    print(f"Run dir: {run_dir}")
    print(f"Final mean reward: {final_stats['mean_reward']:.2f} ± {final_stats['std_reward']:.2f}")


if __name__ == "__main__":
    main()
