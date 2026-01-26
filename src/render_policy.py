import os
import argparse
import pickle
import numpy as np
import gymnasium as gym
import neat

from stable_baselines3 import DQN
from utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, choices=["dqn", "neat"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/videos")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=args.out_dir, episode_trigger=lambda ep: True)

    obs, _ = env.reset(seed=args.seed)

    if args.type == "dqn":
        model = DQN.load(args.model_path)

        def act(obs):
            a, _ = model.predict(obs, deterministic=True)
            return int(a)
    else:
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

        def act(obs):
            out = net.activate(obs)
            return int(np.argmax(out))

    done = False
    total_r = 0.0
    while not done:
        action = act(obs)
        obs, r, terminated, truncated, _ = env.step(action)
        total_r += float(r)
        done = terminated or truncated

    env.close()
    print(f"Video saved to: {args.out_dir}")
    print(f"Episode reward: {total_r:.1f}")


if __name__ == "__main__":
    main()
