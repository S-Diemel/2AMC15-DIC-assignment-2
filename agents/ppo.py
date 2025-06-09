import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Importing BaseAgent class
from agents import BaseAgent

def set_all_seeds(seed: int):
    """Since we want all results to be reproducible we need to set random seeds for all random modules we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_gae(rewards, values, next_value, terminated, gamma, lam):
    """Using Generalized Advantage Estimation to calculate advantage"""
    # Convert to np array to make it faster
    rewards = np.array(rewards)
    values = np.array(values + [float(next_value)])
    terminated = np.array(terminated, dtype=np.float32)

    advantages = np.zeros_like(rewards)
    gae = 0.0
    # Calculation of GAE based on PPO paper 2017
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - terminated[step]) - values[step] # If done, next_value is 0, since terminated
        gae = delta + gamma * lam * (1 - terminated[step]) * gae    # Keep bootstrapping if truncated
        advantages[step] = gae

    returns = advantages + values[:-1]
    return advantages, returns

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
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class RolloutBuffer:
    """PPO works with a rollout buffer, storing data for n_steps before updating"""
    def __init__(self):
        self.clear()

    def add(self, state=None, action=None, reward=None, log_prob=None, value=None, terminated=None):
        if state is not None:
            self.states.append(state)
        if action is not None:
            self.actions.append(action)
        if reward is not None:
            self.rewards.append(reward)
        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)
        if terminated is not None:
            self.terminated.append(terminated)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.terminated = []


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
        rollout_steps=4096,
        num_envs=4
    ):
        super().__init__()
        set_all_seeds(seed)
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
        self.num_envs = num_envs

        # Policy and value network based on actor-critic network
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.policy_old = ActorCritic(state_size, action_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()
        self.buffer = [RolloutBuffer() for _ in range(num_envs)]
        self.step_counter = 0

    def take_action_training(self, state: tuple[int, int] | np.ndarray, env_idx: int) -> int:
        """Returns actions for given state as per current policy."""
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():  # No gradients needed for action selection
            probs, value = self.policy_old(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        # Add action, log_pop, value to buffer
        return action.item(), log_prob.item(), value.item()

    def take_action(self, state: tuple[int, int] | np.ndarray) -> int:
        """Returns greedy action for eval"""
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, _ = self.policy_old(state_tensor)  # Ignore value estimate during eval
            greedy_action = torch.argmax(probs).item()

        return greedy_action

    def calc_advantages_for_trajectory(self, env_buffer: RolloutBuffer):
        values = env_buffer.values
        rewards = env_buffer.rewards
        terminated = env_buffer.terminated

        # If episode ended before iters
        if terminated[-1]:  # Episode ended
            next_value = 0.0

        # Find next values for advantage calculation
        with torch.no_grad():
            next_state = torch.tensor(env_buffer.states[-1], dtype=torch.float32).unsqueeze(0).to(self.device)
            next_value = self.policy_old(next_state)[1].squeeze()

        # Advantage calculation, using gae
        advantages, returns = compute_gae(rewards, values, next_value, terminated, self.gamma, self.gae_lambda)

        return advantages, returns

    def create_DL(self, states, actions, old_log_probs, returns, advantages):
        states = torch.tensor(np.concatenate(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.concatenate(actions), dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(np.concatenate(old_log_probs), dtype=torch.float32).to(self.device)
        advantages = torch.tensor(np.concatenate(advantages), dtype=torch.float32).to(self.device)
        returns = torch.tensor(np.concatenate(returns), dtype=torch.float32).to(self.device)

        # Normalize advantages to control variance and stability for PPO update
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create DataLoader for mini-batch updates
        dataset = TensorDataset(states, actions, old_log_probs, returns, advantages)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return dataloader

    def learn(self):
        """Learn from rollout buffer when step>=rollout_steps"""

        advantages, returns = [], []
        states, actions, old_log_probs = [], [], []

        for env_idx in range(self.num_envs):
            env_advantages, env_returns = self.calc_advantages_for_trajectory(self.buffer[env_idx])
            advantages.append(env_advantages)
            returns.append(env_returns)
            states.append(self.buffer[env_idx].states)
            actions.append(self.buffer[env_idx].actions)
            old_log_probs.append(self.buffer[env_idx].log_probs)

        dataloader = self.create_DL(states, actions, old_log_probs, returns, advantages)

        # PPO update for ppo_epochs, mini batch updates using DL
        for epoch in range(self.ppo_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                # Get current policy outputs
                probs, state_values = self.policy(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Compute ratios
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate loss, based on PPO paper
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss using clipped update
                old_values = self.policy_old(batch_states)[1].detach()  # Old value estimates
                clipped_values = old_values + torch.clamp(
                    state_values - old_values,
                    -self.eps_clip,
                    self.eps_clip
                )
                # Loss update based on PPO paper, choosing max loss to be conservative
                critic_loss = torch.max(
                    self.mse_loss(state_values.squeeze(), batch_returns),
                    self.mse_loss(clipped_values.squeeze(), batch_returns)
                )

                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                # Update NN
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)  # Gradient clipping
                self.optimizer.step()

        # Copy new policy to old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffers for all envs
        for env_idx in range(self.num_envs):
            self.buffer[env_idx].clear()

    def update(self, state, action, reward, log_prob, value, terminated, env_idx):
        """Add experience to buffer"""

        self.buffer[env_idx].add(state=state, action=action, reward=reward, log_prob=log_prob, value=value, terminated=terminated)
        self.step_counter += 1

        if self.step_counter >= self.rollout_steps:
            self.learn()
            self.step_counter = 0

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