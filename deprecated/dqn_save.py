import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Import the BaseAgent class
from agents import BaseAgent


def set_all_seeds(seed: int):
    """Since we want all results to be reproducible we need to set random seeds for all random modules we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_dims=(64, 64)):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = state_size
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        set_all_seeds(seed)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.from_numpy(np.vstack(states)).float(),
            torch.from_numpy(np.array(actions)).long(),
            torch.from_numpy(np.array(rewards)).float(),
            torch.from_numpy(np.vstack(next_states)).float(),
            torch.from_numpy(np.array(dones).astype(np.uint8)).float(),
        )

    def __len__(self):
        return len(self.memory)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        seed: int = 0,
        buffer_size: int = int(1e5),
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 1e-3,
        update_every: int = 4,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super(DQNAgent, self).__init__()
        set_all_seeds(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size).to(self._device())
        self.qnetwork_target = QNetwork(state_size, action_size).to(self._device())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        self.t_step = 0

        # Epsilon for epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def take_action(self, state: tuple[int, int] | np.ndarray) -> int:
        """
        Returns actions for given state as per current policy.
        """
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self._device())
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return int(torch.argmax(action_values.cpu()).item())
        else:
            return random.choice(np.arange(self.action_size))

    def update(self, state, reward, action):
        """
        This method should be called AFTER each step in the environment
        to store transitions and potentially trigger learning.

        Args:
            state: next state (after the action)
            reward: float reward from the environment
            action: int action that was taken
        """
        # Here, 'state' passed is next_state; environment code uses state after step
        # We need to store (prev_state, action, reward, next_state, done)
        # For simplicity, we manage prev_state internally
        if not hasattr(self, 'prev_state'):
            self.prev_state = None
            self.prev_action = None
        
        # On first call, only store state for next time
        if self.prev_state is None:
            self.prev_state = state
            self.prev_action = action
            return

        # Otherwise, sample done flag as False; the environment signals terminal separately
        done = False

        # Add to replay buffer
        self.memory.add(self.prev_state, self.prev_action, reward, state, done)
        self.prev_state = state
        self.prev_action = action

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        """
        states, actions, rewards, next_states, dones = experiences
        device = self._device()
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - dones.unsqueeze(1)))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network -------------------
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def soft_update(self, local_model, target_model):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

