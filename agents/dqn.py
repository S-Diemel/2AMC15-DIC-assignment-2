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
    """
    Initializes a simple Q-network with two hidden layers of 64 neurons each. 
    Lecturer said that it is good to keep it simple, for our simple task.
    """
    def __init__(self, state_size: int, action_size: int, hidden_dims=(128, 128)):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(  # Initializing a simple MLP to approximate Q-values
            nn.Linear(state_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    

class ReplayBuffer:
    """Replay buffer which is an essential part of DQN agent."""
    def __init__(self, buffer_size: int, batch_size: int, seed: int = 0):
        self.memory = deque(maxlen=buffer_size)  # First in first out buffer with some max length
        self.batch_size = batch_size
        set_all_seeds(seed)  # important to make sure that batch sampling is reproducible

    def add(self, state, action, reward, next_state, done):
        """This method adds a new transition to the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """This method samples a batch of transitions from the replay buffer."""
        batch = random.sample(self.memory, k=self.batch_size)  # Sample from memory
        states, actions, rewards, next_states, terminal_next_state = zip(*batch)  # a clean way to unpack the batches into states, actions, etc.
        return (
            torch.from_numpy(np.vstack(states)).float(),
            torch.from_numpy(np.array(actions)).long(),
            torch.from_numpy(np.array(rewards)).float(),
            torch.from_numpy(np.vstack(next_states)).float(),
            torch.from_numpy(np.array(terminal_next_state).astype(np.uint8)).float(),
        )  # making sure everything is returned in the right format for the Q-network

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
        # tau no longer needed for hard updates
        target_update_every: int = 5000,  # 10000 more usual, but since small space, likely fine
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
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())  # initialize target network with local network weights

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

    def take_action(self, state: tuple[int, int] | np.ndarray) -> int:
        """
        Returns actions for given state as per current policy.
        """
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self._device())
        self.qnetwork_local.eval()  # Set the network to evaluation mode, to only obtain action and not save gradients
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

        Args:
            state: next state (after the action)
            reward: float reward from the environment
            action: int action that was taken
        """        
        done = terminated

        # Add to replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.warmup_start_steps:  
            # start learning weights when memory contains enough samples to sample a batch
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        """
        # Note that the code below is done in parallel for the whole batch
        states, actions, rewards, next_states, terminal_next_state = experiences
        device = self._device()
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        terminal_next_state = terminal_next_state.to(device)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)  # returns the value Q value of the best (next_)action in the next state
        # Compute Q targets for current states
        Q_targets = rewards.unsqueeze(1) + (self.gamma * Q_targets_next * (1 - terminal_next_state.unsqueeze(1)))
        # (1 - terminal_next_state.unsqueeze(1)) makes sure that if next state is terminal, we do not add future rewards, but only immediate reward.
        # But we only want to do this if the target state is reached, not if episode ended by iteration limit.
        # TODO: check final choice above for termination flag

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1))  # gets the Q value for the action taken in the current state, from the local network

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)

        # Tracking the loss a little bit:
        # if self.learn_steps % 500 == 0:
        #     print(f"Step {self.learn_steps} - Q-loss = {loss.item():.4f}")

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Hard update target network every self.target_update_every steps
        self.learn_steps += 1
        if self.learn_steps % self.target_update_every == 0:
            #print("Target network is updated")
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
