"""Train a CartPole-v1 DQN with soft target-network updates."""

import argparse
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReplayBuffer:
    """Keep recent transitions and sample unrelated training examples."""

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminals = zip(
            *transitions, strict=True
        )
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(terminals, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQN(nn.Module):
    """Estimate one Q-value per action from the four CartPole measurements."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "plots")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    online_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    learning_rate = 0.00025
    gamma = 0.99
    batch_size = 64
    memory = ReplayBuffer(50000)
    tau = 0.01
    optimizer = optim.Adam(online_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    rewards_per_episode = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"CartPole-v1 DQN v2 | episodes={args.episodes} | seed={args.seed}",
        flush=True,
    )
    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        total_reward = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = online_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            # Store only termination in the target mask, not a time-limit stop.
            memory.push(state, action, reward, next_state, terminated)
            total_reward += reward
            state = next_state

            if len(memory) > batch_size:
                states, actions, rewards, next_states, terminals = memory.sample(
                    batch_size
                )
                current_q = (
                    online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                )
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - terminals)

                loss = loss_fn(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Blend the target weights without recording gradients.
                with torch.no_grad():
                    for target_param, online_param in zip(
                        target_net.parameters(), online_net.parameters(), strict=True
                    ):
                        target_param.copy_(
                            tau * online_param + (1.0 - tau) * target_param
                        )

            if terminated or truncated:
                break

        epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * episode)
        rewards_per_episode.append(total_reward)
        if (episode + 1) % 50 == 0:
            mean_reward = np.mean(rewards_per_episode[-50:])
            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"last 50 mean reward={mean_reward:.2f}",
                flush=True,
            )

    env.close()
    window = min(25, len(rewards_per_episode))
    moving_average = np.convolve(
        rewards_per_episode, np.ones(window) / window, mode="valid"
    )
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_per_episode, alpha=0.3, color="blue", label="Episode reward")
    plt.plot(
        range(window - 1, len(rewards_per_episode)),
        moving_average,
        color="green",
        label=f"{window}-episode moving average",
    )
    plt.title(f"CartPole DQN v2: soft updates | seed {args.seed}")
    plt.xlabel("Episode")
    plt.ylabel("Total reward (survival steps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output = args.output_dir / "cartpole_rewards_v2.png"
    plt.savefig(output)
    plt.close()
    capped_episodes = sum(reward == 500 for reward in rewards_per_episode)
    print(
        f"Complete | last 50 mean={np.mean(rewards_per_episode[-50:]):.2f} | "
        f"max reward={max(rewards_per_episode):.0f} | "
        f"500-step episodes={capped_episodes} | plot={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
