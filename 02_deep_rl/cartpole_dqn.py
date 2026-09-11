import collections
import os
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.makedirs("../plots", exist_ok=True)


class ReplayBuffer:
    def __init__(self, capacity):
        # deque automatically removes old items when capacity is reached
        self.buffer = collections.deque(maxlen=capacity)

    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        # Pick random transitions to break correlation between consecutive frames
        transitions = random.sample(self.buffer, batch_size)
        states,actions,rewards,next_states,dones = zip(*transitions)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

#Deep Q-Network (The Estimator Network Architecture)
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Outputs Q-values for each action
        )

    def forward(self, x):
        return self.net(x)

#Initialize Environment & Networks
env = gym.make("CartPole-v1")  
state_dim = env.observation_space.n if hasattr(env.observation_space, 'n') else env.observation_space.shape[0]  
action_dim = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

print(f"State boyutları (Giriş katmanı için): {state_dim}")    # CartPole için 4 basacaktır
print(f"Action boyutları (Çıkış katmanı için): {action_dim}")  # CartPole için 2 basacaktır

# Two identical neural networks (Secret Weapon #2)
online_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim,action_dim)

# Initialize target network with the exact same weights and freeze it
target_net.load_state_dict(online_net.state_dict())
target_net.eval()

# Hyperparameters
lr = 0.001
gamma = 0.99
batch_size = 64
memory_capacity = 10000
target_update_frequency = 1000 # How often to copy weights to target network
total_episodes = 500

memory = ReplayBuffer(memory_capacity)
optimizer = optim.Adam(online_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()

# Epsilon-Greedy parameters
epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

rewards_per_episode = []
global_step = 0

print("CartPole DQN Training is starting...")

#Training Loop
for episode in range(total_episodes):
    state, info = env.reset()
    total_reward = 0

    while True:
        global_step += 1
        # Action Selection using Epsilon-Greedy
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
              # Predict Q-values and pick the index of the highest one (argmax)  
                action = online_net(state_tensor).argmax().item()


        next_state, reward, done, truncated, info = env.step(action)

        is_terminal = done or truncated
        # Push this step to our Experience Replay memory 
        memory.push(state,action,reward,next_state,is_terminal)

        total_reward += reward
        state = next_state

        # Optimization Step (Train neural net if we have enough memory samples)
        if len(memory) > batch_size:
            states, actions, rewards, next_states, dones = memory.sample(batch_size)

            # --- BELLMAN ALIGNED LOSS COMPUTATION ---
            # Current Prediction: What our Online Net thinks this action is worth

            current_q = online_net(states).gather(1,actions.unsqueeze(1)).squeeze(1)

             # Target Calculation: What our Frozen Target Net predicts for the next state
            
            with torch.no_grad():
                max_next_q = target_net(next_states).max(1)[0]

                target_q = rewards + gamma * max_next_q * (1 - dones)

            loss = loss_fn(current_q, max_next_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % target_update_frequency == 0:
            target_net.load_state_dict(online_net.state_dict())

        if is_terminal:
            break

    epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * episode)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(rewards_per_episode[-50:])
        print(f"Episode: {episode + 1}/{total_episodes} | 50-Ep Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
    
print("DQN Training completed successfully!")

# 5. Stability Analysis & Plotting
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, color="blue", label="Raw Episode Reward")
moving_avg = np.convolve(rewards_per_episode, np.ones(25)/25, mode='valid')
plt.plot(range(24, total_episodes), moving_avg, color="red", linewidth=2, label="25-Ep Moving Average")
plt.title("CartPole DQN Training Stability")
plt.xlabel("Episodes")
plt.ylabel("Total Reward (Survival Steps)")
plt.legend()
plt.grid(True)
plt.savefig("../plots/cartpole_rewards.png")
print("Training plot saved inside 'plots/cartpole_rewards.png'!")






