# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/114TIjIxp7bFdr2lR-KhomWZr7HqI_Ke2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_ratio=0.2, epochs=10):
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.critic_net = CriticNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.critic_net.parameters()), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.buffer = ReplayBuffer()

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        mu, std = self.policy_net(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        value = self.critic_net(state)
        return action.cpu().numpy(), log_prob, value

    def store_transition(self, state, action, reward, log_prob, value):
        self.buffer.store(state, action, reward, log_prob, value)

    def update(self, next_state):
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        next_value = self.critic_net(next_state).detach()
        self.buffer.compute_advantages(self.gamma, self.lam, next_value)

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(device)
        returns = torch.tensor(self.buffer.returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32).to(device)
        old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32).to(device)

        for _ in range(self.epochs):
            mu, std = self.policy_net(states)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Surrogate objective with clipping (Equation 7)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
            policy_loss = -surrogate_loss.mean()

            # Critic loss
            values = self.critic_net(states).squeeze()
            value_loss = ((returns - values) ** 2).mean()

            # Total loss (Equation 9)
            loss = policy_loss + 0.5 * value_loss

            # Optimize the combined loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear the buffer after updating
        self.buffer.clear()