import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt




class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.mu = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mu = self.mu(x)  # Mean of the Gaussian distribution
        log_std = torch.clamp(self.log_std, -20, 2)  # Optional: clamp log_std to avoid extreme values
        std = torch.exp(log_std)  # Convert log_std to standard deviation
        return mu, std



# Actor Network (Policy Network)
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.mu = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mu = self.mu(x)  # Mean of the Gaussian distribution
        log_std = torch.clamp(self.log_std, -20, 2)  # Optional: clamp log_std to avoid extreme values
        std = torch.exp(log_std)  # Convert log_std to standard deviation
        return mu, std


class CriticNetwork(nn.Module):
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 64)        # Second hidden layer
        self.value = nn.Linear(64, 1)       # Output layer: scalar value

    def forward(self, state):
        x = torch.relu(self.fc1(state))    # Apply ReLU to the first layer
        x = torch.relu(self.fc2(x))        # Apply ReLU to the second layer
        value = self.value(x)              # Scalar value output
        return value  # Return scalar value for the state




class ReplayBuffer:
    def __init__(self, batch_size=10000):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []
        self.batch_size = batch_size

    def push(self, state, action, reward, reward_togo, advantage, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.rewards_togo.append(reward_togo)
        self.advantages.append(advantage)
        self.values.append(value)  
        self.log_probs.append(log_prob)

    def sample(self):
        num_states = len(self.states)
        batch_start = torch.arange(0, num_states, self.batch_size)
        indices = torch.randperm(num_states)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return (torch.tensor(self.states), 
                torch.tensor(self.actions), 
                torch.tensor(self.rewards),
                torch.tensor(self.rewards_togo),
                torch.tensor(self.advantages),
                torch.tensor(self.values),
                torch.tensor(self.log_probs), 
                batches)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.rewards_togo = []
        self.advantages = []
        self.values = []
        self.log_probs = []

