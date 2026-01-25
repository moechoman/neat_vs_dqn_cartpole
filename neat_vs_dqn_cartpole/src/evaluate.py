import os
import argparse
import pickle
import yaml
import neat
import numpy as np

from stable_baselines3 import DQN

from utils import save_json, evaluate_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--type", type=str, choices=["dqn", "neat"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="eval.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    env_id = cfg["env_id"]
    eval_eps = int(cfg["budget"]["eval_episodes"])

    if args.type == "dqn":
        model = DQN.load(args.model_path)

        def act_fn(obs):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

    else:
        # NEAT winner pickle + need NEAT config to build net
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "configs/neat.cfg",
        )
        with open(args.model_path, "rb") as f:
            winner = pickle.load(f)

        net = neat.nn.FeedForwardNetwork.create(winner, neat_config)

        def act_fn(obs):
            out = net.activate(obs)
            return int(np.argmax(out))

    stats = evaluate_policy(env_id=env_id, act_fn=act_fn, seed=args.seed + 999, n_episodes=eval_eps)
    save_json(args.out, stats)
    print(stats)


if __name__ == "__main__":
    main()
