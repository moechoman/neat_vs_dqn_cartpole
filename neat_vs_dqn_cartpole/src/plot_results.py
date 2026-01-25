import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import ensure_dir, load_json


def load_progress_csvs(root_dir: str, algo: str):
    """
    Returns list of (progress_df, final_eval_dict)
    """
    runs = sorted(glob.glob(os.path.join(root_dir, algo, "*")))
    all_runs = []
    for run in runs:
        prog_path = os.path.join(run, f"{algo}_progress.csv")
        final_path = os.path.join(run, "final_eval.json")
        if os.path.exists(prog_path) and os.path.exists(final_path):
            df = pd.read_csv(prog_path)
            final = load_json(final_path)
            all_runs.append((df, final, run))
    return all_runs


def interpolate_curve(x, y, x_grid):
    # If too short, return NaNs
    if len(x) < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    return np.interp(x_grid, x, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--out_dir", type=str, default="results/plots")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    dqn_runs = load_progress_csvs(args.results_dir, "dqn")
    neat_runs = load_progress_csvs(args.results_dir, "neat")

    # ---------- Learning curves ----------
    # Build a common x-grid in env steps
    x_grid = np.linspace(0, 200000, 101)

    def aggregate_learning(runs, algo):
        Ys = []
        for df, _, _ in runs:
            if algo == "dqn":
                x = df["env_steps"].values
                y = df["mean_reward_20ep"].values
            else:
                x = df["env_steps_used"].values
                y = df["best_fitness"].values
            Ys.append(interpolate_curve(x, y, x_grid))
        Y = np.vstack(Ys) if len(Ys) > 0 else np.empty((0, len(x_grid)))
        return Y

    Y_dqn = aggregate_learning(dqn_runs, "dqn")
    Y_neat = aggregate_learning(neat_runs, "neat")

    plt.figure()
    if Y_dqn.shape[0] > 0:
        m = np.nanmean(Y_dqn, axis=0)
        s = np.nanstd(Y_dqn, axis=0)
        plt.plot(x_grid, m, label="DQN")
        plt.fill_between(x_grid, m - s, m + s, alpha=0.2)

    if Y_neat.shape[0] > 0:
        m = np.nanmean(Y_neat, axis=0)
        s = np.nanstd(Y_neat, axis=0)
        plt.plot(x_grid, m, label="NEAT")
        plt.fill_between(x_grid, m - s, m + s, alpha=0.2)

    plt.xlabel("Environment Steps")
    plt.ylabel("Reward (DQN eval mean / NEAT best fitness)")
    plt.title("Learning Curves (Mean ± Std over seeds)")
    plt.legend()
    plt.grid(True)

    curve_path = os.path.join(args.out_dir, "learning_curves.png")
    plt.savefig(curve_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- Final performance ----------
    dqn_final = [final["mean_reward"] for _, final, _ in dqn_runs]
    neat_final = [final["mean_reward"] for _, final, _ in neat_runs]

    plt.figure()
    labels = ["DQN", "NEAT"]
    means = [np.mean(dqn_final), np.mean(neat_final)]
    stds = [np.std(dqn_final), np.std(neat_final)]
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.ylabel("Final Mean Reward (100 eval eps)")
    plt.title("Final Performance (Mean ± Std over seeds)")
    plt.grid(True, axis="y")

    final_path = os.path.join(args.out_dir, "final_performance.png")
    plt.savefig(final_path, dpi=200, bbox_inches="tight")
    plt.close()

    # ---------- Summary table ----------
    rows = []
    for algo, runs in [("DQN", dqn_runs), ("NEAT", neat_runs)]:
        for df, final, run in runs:
            rows.append({
                "algo": algo,
                "run": os.path.basename(run),
                "final_mean_reward": final["mean_reward"],
                "final_std_reward": final["std_reward"],
                "train_time_sec": final.get("train_time_sec", np.nan),
                "env_steps_used": final.get("env_steps_used", final.get("max_env_steps", np.nan)),
            })

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print("Saved plots:")
    print(" -", curve_path)
    print(" -", final_path)
    print("Saved summary CSV:", summary_path)


if __name__ == "__main__":
    main()
