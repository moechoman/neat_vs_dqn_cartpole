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

    # Mode controls whether we use a trained model or a forced failing policy
    parser.add_argument(
        "--mode",
        type=str,
        default="model",
        choices=["model", "random", "left", "right"],
        help="What to render: trained model or a failing policy.",
    )

    # Only needed when mode == "model"
    parser.add_argument("--type", type=str, choices=["dqn", "neat"], default=None)
    parser.add_argument("--model_path", type=str, default=None)

    parser.add_argument("--out_dir", type=str, default="results/videos")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode (CartPole max is 500).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")


    args = parser.parse_args()

    ensure_dir(args.out_dir)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args.out_dir,
        episode_trigger=lambda ep: True,  # record every episode
    )

    obs, _ = env.reset(seed=args.seed)
    # Force near-failure starting state (more dramatic)
    env.unwrapped.state = np.array([0.0, 0.0, 0.20, 0.0], dtype=np.float32)

    # -------------------------------
    # Build action function (act)
    # -------------------------------
    if args.mode == "model":
        if args.type is None or args.model_path is None:
            raise ValueError("When --mode model, you must provide --type and --model_path.")

        if args.type == "dqn":
            model = DQN.load(args.model_path)

            def act(o):
                a, _ = model.predict(o, deterministic=True)
                return int(a)

        elif args.type == "neat":
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

            def act(o):
                out = net.activate(o)
                return int(np.argmax(out))

        else:
            raise ValueError(f"Unknown type: {args.type}")

    else:
        # Failing / baseline policies
        def act(_o):
            if args.mode == "random":
                return env.action_space.sample()
            if args.mode == "left":
                return 0
            if args.mode == "right":
                return 1
            raise ValueError(f"Unknown mode: {args.mode}")

    # -------------------------------
    # Run one episode and record it
    # -------------------------------
    total_r = 0.0
    done = False

    for ep in range(int(args.episodes)):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r = 0.0
        done = False

        for _ in range(int(args.steps)):
            action = act(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += float(r)
            done = bool(terminated or truncated)
            if done:
                break

    print(f"Episode {ep+1}/{args.episodes} reward: {total_r:.1f}")


    env.close()

    print(f"Video saved to: {args.out_dir}")
    print(f"Episode reward: {total_r:.1f}")


if __name__ == "__main__":
    main()
