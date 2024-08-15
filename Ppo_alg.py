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
    deltas = rewards + gamma * values[1:] - values[:-1]
    advantages = np.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lambda_ * gae
        advantages[t] = gae

    return advantages



def update_policy(policy_network, value_network, states, actions, returns, advantages, log_probs_old, clip_param=0.2, epochs=4, lr=1e-3):

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32)


    optimizer_policy = Adam(policy_network.parameters(), lr=lr)
    optimizer_value = Adam(value_network.parameters(), lr=lr)

  
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        
        logits = policy_network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        
        ratios = torch.exp(log_probs - log_probs_old)

        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - clip_param, 1 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

     
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

      
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

        if episode % 10 == 0:
            print(f"Episode {episode}: Total Reward = {env.total_reward}")


env = AmazonWarehouseEnv()
input_dim = env.observation_space.shape[0]  
output_dim = env.action_space.n

policy_network = PolicyNetwork(input_dim, output_dim)
value_network = ValueNetwork(input_dim)

# Train the model
train(env, policy_network, value_network)
