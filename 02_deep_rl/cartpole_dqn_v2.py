import collections
import os
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Fix path to save v2 plots inside the root 'plots' folder
os.makedirs("../plots", exist_ok=True)

# 1. Experience Replay Buffer (Memory)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))
                
    def __len__(self):
        return len(self.buffer)

# 2. Deep Q-Network (The Estimator Network Architecture)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# 3. Initialize Environment & Networks
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0] # Extracted directly as integer
action_dim = env.action_space.n            

online_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

# v2 HYPERPARAMETERS 
lr = 0.00025                 # Hypothesis 2: Lower learning rate to prevent catastrophic forgetting
gamma = 0.99
batch_size = 64
memory_capacity = 50000      # Hypothesis 3: Larger buffer capacity to retain good older memories
total_episodes = 500
tau = 0.01                   # Hypothesis 1: Soft update parameter (Polyak Averaging - 1% blend per step)

memory = ReplayBuffer(memory_capacity)
optimizer = optim.Adam(online_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Epsilon-Greedy parameters
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

rewards_per_episode = []

print("CartPole DQN v2 (Stabilized Version) Training is starting...")

# 4. Core Training Loop
for episode in range(total_episodes):
    state, info = env.reset()
    total_reward = 0
    
    while True:
        # Action Selection using Epsilon-Greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = online_net(state_tensor).argmax().item()
                
        # Take step in the environment
        next_state, reward, done, truncated, info = env.step(action)
        is_terminal = done or truncated
        
        # Push this step to our larger Experience Replay memory
        memory.push(state, action, reward, next_state, is_terminal)
        
        total_reward += reward
        state = next_state
        
        # Optimization Step
        if len(memory) > batch_size:
            states, actions, rewards, next_states, dones = memory.sample(batch_size)
            
            # Current Prediction
            current_q = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Calculation using the Target Network
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]
                target_q = rewards + gamma * max_next_q * (1 - dones)
                
            # Compute difference (Loss) and update Online Net
            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # SOFT UPDATE IMPLEMENTATION (POLYAK AVERAGING) 
            # Instead of copying weights every 1000 steps, we smoothly blend 1% of online weights into target weights every single step.
            for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
                
        if is_terminal:
            break
            
    # Exponential Epsilon Decay
    epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_per_episode.append(total_reward)
    
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(rewards_per_episode[-50:])
        print(f"Episode: {episode + 1}/{total_episodes} | 50-Ep Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")

print(" DQN v2 Training completed successfully!")

#  Analysis & Plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, color="blue", label="Raw Episode Reward")
moving_avg = np.convolve(rewards_per_episode, np.ones(25)/25, mode='valid')
plt.plot(range(24, total_episodes), moving_avg, color="green", linewidth=2, label="25-Ep Moving Average") # Green line for v2 success
plt.title("CartPole DQN v2 Training Stability (Soft Update)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward (Survival Steps)")
plt.legend()
plt.grid(True)
plt.savefig("../plots/cartpole_rewards_v2.png")
print("Training plot saved inside 'plots/cartpole_rewards_v2.png'!")
