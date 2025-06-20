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
        self.shared = nn.Sequential(  # create shared feature representation of state for both actor and critic
            nn.Linear(state_size, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.actor = nn.Sequential(  # Separate actor head for computing the probability distribution across actions
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)  # Separate critic head computing a single value per state

    def forward(self, x):
        x = self.shared(x)
        actor_output = self.actor(x)  # Compute the probability distribution over the actions given a state
        critic_output = self.critic(x).squeeze(-1)  # Ensure critic outputs are squeezed
        # Critic computes a value for each state forming the baseline which we use to compute the advantage of a certain action given a state
        return actor_output, critic_output


class RolloutBuffer:
    """
    Single buffer for all environments with shape (n_steps, n_envs, ...).

    PPO fills buffer till n_steps, then learns. It then does a new rollout but does not recall from
    the previous rollout. So you do not remember old experiences, you only use experiences obtained
    from the last policy to evaluate the current policy. This keeps the gradient estimates unbiased
    for the policy you are optimizing. It avoids the stability and divergence issues, experienced by reusing
    old experiences. 
    """
    def __init__(self, rollout_steps, num_envs, state_size):
        # Rollout configuration parameters
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.state_size = state_size

        # Pre-allocation of numpy arrays for efficiency
        self.states = np.zeros((rollout_steps, num_envs, state_size), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.bool_)

        # Position pointer --> index in pre-allocated array to write the experience to
        self.pos = 0
        # Indicates when the buffer has collected rollout_steps of data
        self.full = False

    def add(self, state, action, reward, log_prob, value, dones):
        """Store one timestep of data for all environments in parallel."""
        # Write each component into the buffer at current position
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.log_probs[self.pos] = log_prob
        self.values[self.pos] = value
        self.dones[self.pos] = dones

        # Move the pointer by one
        self.pos += 1
        # Check if we completed the rollout
        if self.pos == self.rollout_steps:
            # Indicates that the rollout is complete 
            self.full = True

    def clear(self):
        """Resets the buffer for the next batch."""
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

        if seed is not None:
            # Set all seeds, to make sure we have reproducability
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # Use GPU if available to speed up training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set hyperparameters
        self.state_size = state_size  # Dimensionality of state space
        self.action_size = action_size  # Number of possible actions
        self.gamma = gamma  # Discounting factor
        self.lr = lr  # Learning rate for optimizer
        self.eps_clip = eps_clip  # Clipping for trust region in loss function of actor (surrogate loss)
        self.gae_lambda = gae_lambda  # lambda parameter for Gneralized Advantage Estimation
        self.entropy_coef = entropy_coef  # coefficient for entropy which promotes exploration
        self.ppo_epochs = ppo_epochs  # Number of epochs to run each PPO update
        self.batch_size = batch_size  # Mini-batch size for each update
        self.rollout_steps = rollout_steps  # Number of steps to collect before each update
        self.num_envs = num_envs

        # Define the rollout buffer 
        self.buffer = RolloutBuffer(rollout_steps, num_envs, state_size)

        # Policy and value network based on actor-critic network
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.policy_old = ActorCritic(state_size, action_size).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Defining the loss for the critic and the optimizer for the policy Actor Critic network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()

    def take_action_training(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns actions for all environments in parallel. However, during training, we do not grab the
        greedy action, but we sample from the action distribution. Furthermore we return everything instead of only the action.
        """
        # Convert state as numpy arrow to a tensor on the GPU
        state_tensor = torch.from_numpy(states).float().to(self.device)
        with torch.no_grad():  # Don't maintain the gradients during a take-action, as we are not updating the network now
            probs, values = self.policy_old(state_tensor)  # probability distributions for all environments and state values for all environments
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()  # Sample an action from the probability distribution over the actions
            log_probs = dist.log_prob(actions)  # Return the log probabilities, useful for computing the gradient of the network

            values = values.squeeze(-1)  # Removes last dimension if it's 1

        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def take_action(self, state: tuple[int, int] | np.ndarray) -> int:
        """Returns greedy action for evaluation."""
        # Convert state list representation state to numpy array and then to tensor on GPU
        state = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Greedily obtain the action with the highest probability given the state
            probs, _ = self.policy_old(state_tensor)  # Ignore value estimate of the state during evaluation
            greedy_action = torch.argmax(probs).item()

        return greedy_action

    def compute_gae_and_returns(self, last_values, last_dones):
        """
        Compute Generalized Advantage Estimation (GAE) and returns for all environments in parallel.
        
        Parameters:
        - last_values: The value estimates of the final state for each environment.
        - last_dones: A binary indicator (1 or 0) for whether the episode ended in each environment.
        
        Returns:
        - advantages: Advantage estimates for each time step.
        - returns: Computed returns (advantage + baseline value).
        """
        # append last value estimates for each environment at the end of the value array for bootstrap calculation
        values = np.concatenate([self.buffer.values, last_values[np.newaxis, :]], axis=0)
        # Get the rewards from the buffer
        rewards = self.buffer.rewards
        # append last done indicators for each environment at the end of the dones array for bootstrap calculation
        dones = np.concatenate([self.buffer.dones, last_dones[np.newaxis, :]], axis=0)

        # Initialize the advantages array with same shape as rewards
        advantages = np.zeros_like(rewards)
        gae = 0.0  # Initialize GAE value

        # Compute GAE
        for step in reversed(range(self.rollout_steps)):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]  # Temporal difference error
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae  # Update the GAE
            advantages[step] = gae  # Store the computed advantage

        returns = advantages + self.buffer.values  # Advantage plus baseline is the return
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

        # Flatten all buffers
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
        # For each experience add all necessary information to the rollout buffer
        self.buffer.add(states, actions, rewards, log_probs, values, dones)

        if self.buffer.full:  # If the buffer is full we trigger the learning process
            self.learn()

    def save(self, path: str):
        """Save the current model and optimizer state to allow continuation of training."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load the current model and optimizer state to allow continuation of training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(self.policy.state_dict())
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])