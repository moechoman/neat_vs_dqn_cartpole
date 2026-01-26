import subprocess
import yaml
import sys

def main():
    with open("configs/experiment.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    seeds = cfg["seeds"]

    py = sys.executable  # ✅ always uses the uv environment python

    for s in seeds:
        print(f"\n=== SEED {s}: DQN ===")
        subprocess.run([py, "src/train_dqn.py", "--seed", str(s)], check=True)

        print(f"\n=== SEED {s}: NEAT ===")
        subprocess.run([py, "src/train_neat.py", "--seed", str(s)], check=True)

    print("\n=== Plotting Results ===")
    subprocess.run([py, "src/plot_results.py"], check=True)

if __name__ == "__main__":
    main()
