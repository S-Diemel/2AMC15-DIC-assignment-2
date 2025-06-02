import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
lr = 3e-4
gamma = 0.99
lam = 0.95
eps_clip = 0.2
k_epochs = 4
T_horizon = 200

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


def compute_gae(rewards, values, dones, next_value):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    return returns


class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.memory = Memory()

    def take_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs, value = self.policy_old(state)
        dist = Categorical(probs)
        action = dist.sample()

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(dist.log_prob(action))
        self.memory.values.append(value)

        return action.item()

    def update(self):
        # Convert memory to tensors
        states = torch.cat(self.memory.states).to(device)
        actions = torch.stack(self.memory.actions).to(device)
        old_logprobs = torch.stack(self.memory.logprobs).to(device)
        values = torch.cat(self.memory.values).squeeze().detach().cpu().numpy()
        rewards = self.memory.rewards
        dones = self.memory.dones

        # Compute next value for bootstrapping
        with torch.no_grad():
            next_state = self.memory.states[-1]
            _, next_value = self.policy_old(next_state)
            next_value = next_value.item()

        returns = compute_gae(rewards, list(values), dones, next_value)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.tensor(values, dtype=torch.float32).to(device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(k_epochs):
            probs, state_values = self.policy(states)
            dist = Categorical(probs)
            entropy = dist.entropy().mean()
            new_logprobs = dist.log_prob(actions.squeeze())

            ratios = torch.exp(new_logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            loss = -torch.min(surr1, surr2).mean() + \
                   0.5 * self.mse_loss(state_values.squeeze(), returns) - \
                   0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def save(self, path: str):
        """Save the policy network to disk."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        """Load the policy network from disk."""
        self.policy.load_state_dict(torch.load(path))
        self.policy_old.load_state_dict(self.policy.state_dict())  # sync old policy

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()