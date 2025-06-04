import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
lr = 3e-4
gamma = 0.9
lam = 0.95
eps_clip = 0.2
k_epochs = 10
T_horizon = 1000
batch_size = 64
entropy_coef = 0.01


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


def compute_gae(rewards, values, terminated, truncated, next_value, gamma=0.9, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    values = values + [next_value]
    gae = 0
    returns = []

    for step in reversed(range(len(rewards))):
        # For truncated episodes, we should bootstrap from the current value estimate
        # For terminated episodes, we should not bootstrap (value=0)
        non_terminal = 1 - terminated[step]
        non_truncated = 1 - truncated[step]

        delta = rewards[step] + gamma * values[step + 1] * non_terminal - values[step]
        gae = delta + gamma * lam * non_terminal * non_truncated * gae
        returns.insert(0, gae + values[step])
    return returns


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.terminated = []  # Renamed from dones to be more explicit
        self.truncated = []  # New: store truncated flags
        self.values = []

    def clear(self):
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.values.clear()


class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.memory = Memory()

    def take_action(self, state):
        # Store raw state, convert to tensor when needed
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():  # No gradients needed for action selection
            probs, value = self.policy_old(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()
            logprob = dist.log_prob(action)

        # Store raw state instead of tensor
        self.memory.states.append(state)
        self.memory.actions.append(action.item())
        self.memory.logprobs.append(logprob.item())
        self.memory.values.append(value.item())

        return action.item()

    def store_transition(self, reward, terminated, truncated):
        """Store reward and termination flags"""
        self.memory.rewards.append(reward)
        self.memory.terminated.append(terminated)
        self.memory.truncated.append(truncated)

    def update(self, next_state=None):
        # Compute next value for bootstrapping
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                _, next_value = self.policy_old(next_state_tensor)
                next_value = next_value.item()
        else:
            next_value = 0  # Terminal state

        # Compute returns using GAE
        returns = compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.terminated,
            self.memory.truncated,
            next_value,
            gamma,
            lam
        )

        # Convert to tensors
        states = torch.FloatTensor(self.memory.states).to(device)
        actions = torch.LongTensor(self.memory.actions).to(device)
        old_logprobs = torch.FloatTensor(self.memory.logprobs).to(device)
        returns = torch.FloatTensor(returns).to(device)
        values = torch.FloatTensor(self.memory.values).to(device)

        # Compute advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create DataLoader for mini-batch updates
        dataset = TensorDataset(states, actions, old_logprobs, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # PPO update for k epochs
        for epoch in range(k_epochs):
            for batch_states, batch_actions, batch_old_logprobs, batch_returns, batch_advantages in dataloader:
                # Get current policy outputs
                probs, state_values = self.policy(batch_states)
                dist = Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratios
                ratios = torch.exp(new_logprobs - batch_old_logprobs)

                # Clipped surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss using clipped update
                old_values = self.policy_old(batch_states)[1].detach()  # Old value estimates
                clipped_values = old_values + torch.clamp(
                    state_values - old_values,
                    -eps_clip,
                    eps_clip
                )
                # Use the worse (clipped or unclipped) to be conservative
                critic_loss = torch.max(
                    self.mse_loss(state_values.squeeze(), batch_returns),
                    self.mse_loss(clipped_values.squeeze(), batch_returns)
                )

                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
                self.optimizer.step()

        # Copy new policy to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def collect_rollout(self, env, T_horizon, T_truncate):
        state = env.reset_env(no_gui=True)  # Unpack the reset tuple
        total_reward = 0

        for timestep in range(T_horizon):
            action = self.take_action(state)
            next_state, reward, terminated, info = env.step(action)  # Updated to handle truncated

            # Check if truncated
            if timestep >= T_truncate:
                truncated = True
            else:
                truncated = False

            # Optional reward shaping
            if reward == -1:
                reward = -1 * ((abs(next_state[7]) + abs(next_state[8])) / 10)

            # Store both terminated and truncated flags
            self.store_transition(reward, terminated, truncated)
            total_reward += reward
            state = next_state

            if terminated or truncated:
                state = env.reset_env(no_gui=True) # Reset after episode ends
                break
        self.update(state)
        return total_reward

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])