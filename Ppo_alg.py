import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)




def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).

    Parameters:
    - rewards (np.array): Array of rewards.
    - values (np.array): Array of value estimates v(s) for each state, with the
      terminal state appended.
    - gamma (float): Discount factor.
    - lambda_ (float): GAE parameter.

    Returns:
    - advantages (np.array): The estimated advantages.
    """
    deltas = rewards + gamma * values[1:] - values[:-1]
    advantages = np.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lambda_ * gae
        advantages[t] = gae

    return advantages



def update_policy(policy_network, value_network, states, actions, returns, advantages, log_probs_old, clip_param=0.2, epochs=4, lr=1e-3):
    # Convert lists to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)

    # Set up optimizers
    optimizer_policy = Adam(policy_network.parameters(), lr=lr)
    optimizer_value = Adam(value_network.parameters(), lr=lr)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        # Calculate current log probabilities
        logits = policy_network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        # Calculate the ratio between new and old policy
        ratios = torch.exp(log_probs - log_probs_old)

        # Calculate surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Perform policy network update
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Update value network
        values_pred = value_network(states).squeeze()
        value_loss = F.mse_loss(values_pred, returns)

        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()


def train(env, policy_network, value_network, episodes=1000):
    optimizer_policy = optim.Adam(policy_network.parameters(), lr=1e-3)
    optimizer_value = optim.Adam(value_network.parameters(), lr=1e-3)

    for episode in range(episodes):
        state = env.reset()
        done = False

        # Collect data from the environment
        # You need to collect states, actions, rewards, etc.
        
        # After collecting data, compute advantages and update networks
        # This part is left as an exercise

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {env.total_reward}")

# Initializing the environment and networks
env = AmazonWarehouseEnv()
input_dim = env.observation_space.shape[0]  # Adapt this based on your environment
output_dim = env.action_space.n

policy_network = PolicyNetwork(input_dim, output_dim)
value_network = ValueNetwork(input_dim)

# Train the model
train(env, policy_network, value_network)
