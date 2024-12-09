import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
import torch.nn.functional as F



class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        #s=[sin(angle), cos(angle), angular velocity]
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
    def __init__(self, input_size=2, hidden_size=32, output_size=1, num_layers=2, activation=nn.functional.relu):
        super(CriticNetwork, self).__init__()

        # Readout layer
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # RNN layer
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Initialize hidden state with zeros
        self.hidden = torch.zeros(num_layers, hidden_size)

        self.act = activation

    def forward(self, x):

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, self.hidden = self.rnn_layer(x, self.hidden.detach())
        
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        
        return out



class RecurrentPolicyNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, output_size=1, num_layers=2, activation=nn.functional.relu):
        super(RecurrentPolicyNetwork, self).__init__()

        # Readout layer
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # RNN layer
        self.rnn_layer = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Initialize hidden state with zeros
        self.hidden = torch.zeros(num_layers, hidden_size)

        self.act = activation

    def forward(self, x):

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, self.hidden = self.rnn_layer(x, self.hidden.detach())
        
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        
        return out




class ReplayBuffer:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.values = []
        self.log_probs = []
        self.batch_size = batch_size

    def push(self, state, action, reward, advantage, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
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
                torch.tensor(self.advantages),
                torch.tensor(self.values),
                torch.tensor(self.log_probs), 
                batches)
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.values = []
        self.log_probs = []

