import time


# Simulate real-time market data
def get_real_time_market_data():
    # Simulating market fluctuations
    return np.random.uniform(100, 200)


# Real-time strategy execution function
def real_time_trading():
    capital = 10000  # Starting capital
    position = 0  # No initial position
    market_threshold = 0.05  # Adjust strategy if price changes by 5%

    while True:
        current_price = get_real_time_market_data()

        # Adjust the strategy based on the market price change
        if position == 0 and current_price < 150:  # Example: Buy if price is below 150
            position = capital / current_price  # Buy stock
            capital = 0  # Spend all capital
            print(f"BUY at {current_price:.2f}")
        elif position > 0 and current_price > 160:  # Sell if price rises above 160
            capital = position * current_price  # Sell at current price
            position = 0  # Clear position
            print(f"SELL at {current_price:.2f}")

        # Print portfolio status
        print(
            f"Current Capital: {capital:.2f} | Current Position: {position * current_price if position > 0 else 0:.2f}")

        # Sleep for 1 second to simulate real-time execution
        time.sleep(1)


# Start real-time trading
real_time_trading()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.buffer = ReplayBuffer(10000)
        self.gamma = 0.99
        self.batch_size = 64
        self.update_frequency = 100
        self.steps = 0

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())


# Example training loop
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 500

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.add(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode: {episode + 1}, Reward: {total_reward}")

torch.save(agent.model.state_dict(), 'dqn_model.pth')
