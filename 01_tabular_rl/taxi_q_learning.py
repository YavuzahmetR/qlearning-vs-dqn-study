"""Train a Q-table on Taxi-v3 and save its learning curve."""

import argparse
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "plots")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    # Taxi-v3 is available in the pinned Gymnasium 0.29.1.
    env = gym.make("Taxi-v3")
    env.action_space.seed(args.seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    max_steps = 99
    alpha = 0.1
    gamma = 0.6
    epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.005
    rewards_per_episode = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Taxi-v3 | episodes={args.episodes} | seed={args.seed}", flush=True)
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        total_reward = 0

        for _ in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            # A real terminal state has no future return. A time limit does.
            next_max = 0.0 if terminated else np.max(q_table[next_state])
            target = reward + gamma * next_max
            old_value = q_table[state, action]
            q_table[state, action] = old_value + alpha * (target - old_value)

            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * episode)
        rewards_per_episode.append(total_reward)
        if (episode + 1) % 200 == 0:
            mean_reward = np.mean(rewards_per_episode[-200:])
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"last 200 mean reward={mean_reward:.2f}",
                flush=True,
            )

    env.close()
    window = min(50, len(rewards_per_episode))
    moving_average = np.convolve(
        rewards_per_episode, np.ones(window) / window, mode="valid"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, alpha=0.3, color="blue", label="Episode reward")
    plt.plot(
        range(window - 1, len(rewards_per_episode)),
        moving_average,
        color="red",
        label=f"{window}-episode moving average",
    )
    plt.title(f"Taxi-v3 Q-learning | seed {args.seed}")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output = args.output_dir / "taxi_rewards.png"
    plt.savefig(output)
    plt.close()
    print(
        f"Complete | last 200 mean={np.mean(rewards_per_episode[-200:]):.2f} | "
        f"max reward={max(rewards_per_episode):.0f} | plot={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
