import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from agents import BaseAgent


def set_all_seeds(seed: int):
    """Since we want all results to be reproducible we need to set random seeds for all random modules we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QNetwork(nn.Module):
    """
    Initializes a simple Q-network with two hidden layers of 64 neurons each. 
    Lecturer said that it is good to keep it simple, for our simple task.
    """
    def __init__(self, state_size: int, action_size: int, hidden_dims=(128, 128)):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class ReplayBuffer:
    """
    Replay buffer which is an essential part of DQN agent.
    Uses first in first out (FIFO) queue to store transitions.
    Uses a seed for batch sampling to ensure reproducibility.
    """
    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        set_all_seeds(seed)

    def add(self, state, action, reward, next_state, done):
        """This method adds a new transition to the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """This method samples a batch of transitions from the replay buffer."""
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, is_terminal_next_state = zip(*batch)
        return (
            torch.from_numpy(np.vstack(states)).float(),
            torch.from_numpy(np.array(actions)).long(),
            torch.from_numpy(np.array(rewards)).float(),
            torch.from_numpy(np.vstack(next_states)).float(),
            torch.from_numpy(np.array(is_terminal_next_state).astype(np.uint8)).float(),
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
        lr: float = 2.5e-4,
        target_update_every: int = 5000,
        update_every: int = 4,
        epsilon_start: float = 1.0,
        warmup_start_steps: int = 1000,
    ):
        super(DQNAgent, self).__init__()
        set_all_seeds(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every

        # Q-Networks
        self.qnetwork_local = QNetwork(state_size, action_size).to(self._device())
        self.qnetwork_target = QNetwork(state_size, action_size).to(self._device())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        # Initialize target network with local network weights, as required for DQN
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Hard update for target network
        self.target_update_every = target_update_every
        self.learn_steps = 0
 
        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, seed)
        self.t_step = 0
        self.warmup_start_steps = warmup_start_steps

        # Epsilon for epsilon-greedy
        self.epsilon = epsilon_start

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def take_actions_batch(self, states):
        """
        Returns actions for a batch of states as per current policy.
        More efficient than calling take_action individually.
        """
        states = np.array(states, dtype=np.float32)
        state_tensor = torch.from_numpy(states).float().to(self._device())
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection for batch
        actions = []
        for i in range(len(states)):
            if random.random() > self.epsilon:
                actions.append(int(torch.argmax(action_values[i]).item()))
            else:
                actions.append(random.choice(np.arange(self.action_size)))
        return actions

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

    def update(self, state, action, reward, next_state, terminated):
        """
        This method should be called AFTER each step in the environment
        to store transitions and potentially trigger learning.
        """        
        done = terminated
        self.memory.add(state, action, reward, next_state, done)

        # Learn after `update_every` time steps and memory has enough samples
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.warmup_start_steps:  
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        Learning is done on a batch of experiences of fixed size, sampled from the replay buffer.
        """
        states, actions, rewards, next_states, is_terminal_next_state = experiences
        device = self._device()
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        is_terminal_next_state = is_terminal_next_state.to(device)

        # Get max predicted Q values, only update using future rewards if next state not terminal
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        update_using_future_rewards = 1 - is_terminal_next_state.unsqueeze(1)
        Q_targets = rewards.unsqueeze(1) + ((self.gamma * Q_targets_next) * update_using_future_rewards)

        # Gets the Q value for the action taken in the current state, from the local network
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)) 

        # Compute loss and backpropagate
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network params after `self.target_update_every` steps
        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def save(self, path: str):
        """Save the local Q-network’s parameters to disk."""
        torch.save(self.qnetwork_local.state_dict(), path)

    @classmethod
    def load(cls, path: str,
             state_size: int,
             action_size: int,
             seed: int = 0,
             **agent_kwargs):
        """Instantiate a new agent and load saved parameters."""
        agent = cls(state_size, action_size, seed, **agent_kwargs)
        device = agent._device()
        state_dict = torch.load(path, map_location=device)
        agent.qnetwork_local.load_state_dict(state_dict)
        # make sure target matches
        agent.qnetwork_target.load_state_dict(agent.qnetwork_local.state_dict())
        return agent
