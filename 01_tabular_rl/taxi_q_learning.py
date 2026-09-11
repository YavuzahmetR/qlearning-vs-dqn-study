import os
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("../plots", exist_ok=True)


# Initialize Environment
env = gym.make("Taxi-v4")

#Initialize Q-table with zeros(500 states * 6 actions)
state_size = env.observation_space.n if hasattr(env.observation_space, 'n') else env.observation_space.shape[0]
action_size = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]
q_table = np.zeros((state_size,action_size))

# Hyperparameters
total_episodes = 2000
max_steps = 99
alpha = 0.1 # Learning rate
gamma = 0.6 # Discount factor

# Epsilon-Greedy parameters
epsilon = 1.0 # Initial exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005 # decay rate for exploration

rewards_per_episode = []

print("Taxi Q-Learning training is starting...")

# Training Loop
for episode in range(total_episodes):
    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Action selection using Epsilon-Greedy strategy
        if random.uniform(0,1) < epsilon:
            # Explore: choose random action
            action = env.action_space.sample() 
        else:
            # Exploit: choose best known action
            action = np.argmax(q_table[state])

        # Execute the chosen action in the environment
        next_state, reward, done, truncated, info = env.step(action)

        #Bellman Equation
        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state])

        # Q-table memory update
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        total_reward += reward
        state = next_state

        if done or truncated:
            break
    # Epsilon Decay: Reduce exploration rate as the agent learns more about the world        
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    rewards_per_episode.append(total_reward)

    if (episode + 1) % 200 == 0:
        avg_reward = np.mean(rewards_per_episode[-200:])
        print(f"Episode: {episode + 1}/{total_episodes} | 200-Ep Avg Reward: {avg_reward:.2f} | Current Epsilon: {epsilon:.3f}")
print("Training completed successfully!")

plt.figure(figsize=(10,8))
plt.plot(rewards_per_episode, alpha=0.3, color="blue", label = "Raw Episode Reward")


# Calculate 50-episode moving average to smooth out fluctuations and show clear trends
moving_avg = np.convolve(rewards_per_episode, np.ones(50)/50, mode='valid')
plt.plot(range(49, total_episodes), moving_avg, color="red", linewidth=2, label="50-Ep Moving Average")

plt.title("Taxi Q-Learning Training Stability")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.savefig("../plots/taxi_rewards.png")
print("Training plot saved inside 'plots/taxi_rewards.png'!")

