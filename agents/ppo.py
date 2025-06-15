import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Importing BaseAgent class
from agents import BaseAgent

class ActorCritic(nn.Module):
    """
    Simple Actor Critic Neural Network with 1 hidden layer, 64x64
    Small network and 1 hidden layer should be enough for our simple gridword problem
    """
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        actor_output = self.actor(x)
        critic_output = self.critic(x).squeeze(-1)  # Ensure critic outputs are squeezed
        return actor_output, critic_output


class RolloutBuffer:
    """
    Single buffer for all environments with shape (n_steps, n_envs, ...)
    PPO fills buffer till n_steps, then learns
    """
    def __init__(self, rollout_steps, num_envs, state_size):
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.state_size = state_size

        self.states = np.zeros((rollout_steps, num_envs, state_size), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.bool_)
        self.pos = 0
        self.full = False

    def add(self, state, action, reward, log_prob, value, dones):
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.dones[self.pos] = dones

        self.pos += 1
        if self.pos == self.rollout_steps:
            self.full = True

    def clear(self):
        self.pos = 0
        self.full = False


class PPOAgent(BaseAgent):
    def __init__(
        self,
        state_size,
        action_size,
        seed=0,
        gamma=0.99,
        lr=2.5e-4,
        eps_clip=0.2,
        gae_lambda=0.95,
        entropy_coef=0.01,
        ppo_epochs=4,
        batch_size=64,
        rollout_steps=1024,
        num_envs=4
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.eps_clip = eps_clip
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.buffer = RolloutBuffer(rollout_steps, num_envs, state_size)
        self.num_envs = num_envs

        # Policy and value network based on actor-critic network
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.policy_old = ActorCritic(state_size, action_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()
        self.step_counter = 0

    def take_action_training(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns actions for all environments in parallel"""
        state_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():
            probs, values = self.policy_old(state_tensor)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            values = values.squeeze(-1)  # Removes last dimension if it's 1

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.squeeze(-1).cpu().numpy()

    def take_action(self, state: tuple[int, int] | np.ndarray) -> int:
        """Returns greedy action for eval"""
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, _ = self.policy_old(state_tensor)  # Ignore value estimate during eval
            greedy_action = torch.argmax(probs).item()

        return greedy_action

    def compute_gae_and_returns(self, last_values, last_dones):
        """Compute GAE and returns for all environments at once"""
        values = np.concatenate([self.buffer.values, last_values[np.newaxis, :]], axis=0)
        rewards = self.buffer.rewards
        dones = np.concatenate([self.buffer.dones, last_dones[np.newaxis, :]], axis=0)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(self.rollout_steps)):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae

        returns = advantages + self.buffer.values
        return advantages, returns

    def learn(self):
        """Learn from rollout buffer when full"""
        if not self.buffer.full:
            return

        # Get last values for GAE calculation
        with torch.no_grad():
            last_states = torch.tensor(self.buffer.states[-1], dtype=torch.float32).to(self.device)
            last_values = self.policy_old(last_states)[1].squeeze(-1).cpu().numpy()
            last_dones = self.buffer.dones[-1].copy()

        # Compute advantages and returns
        advantages, returns = self.compute_gae_and_returns(last_values, last_dones)

        # Flatten all buffers and normalize advantages
        states = self.buffer.states.reshape(-1, self.state_size)
        actions = self.buffer.actions.flatten()
        old_log_probs = self.buffer.log_probs.flatten()
        returns = returns.flatten()
        advantages = advantages.flatten()

        # Normalize advantages and rewards across all environments and timesteps
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Create DataLoader
        dataset = TensorDataset(
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.long).to(self.device),
            torch.tensor(old_log_probs, dtype=torch.float32).to(self.device),
            torch.tensor(returns, dtype=torch.float32).to(self.device),
            torch.tensor(advantages, dtype=torch.float32).to(self.device)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # PPO update based on the 2017 PPO paper
        for epoch in range(self.ppo_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                # Get current policy outputs
                probs, state_values = self.policy(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Policy loss
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = self.mse_loss(state_values.squeeze(), batch_returns)

                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def update(self, states, actions, rewards, log_probs, values, dones):
        """Add batch of experiences from all environments"""
        self.buffer.add(states, actions, rewards, log_probs, values, dones)

        if self.buffer.full:
            self.learn()

    def save(self, path: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])