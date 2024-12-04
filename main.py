

import gym
import a3_gym_env
import Modules
import collections
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from Modules import CriticNetwork,PolicyNetwork,ReplayBuffer
from torch.distributions import MultivariateNormal

env = gym.make('Pendulum-v1-custom')
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample hyperparameters
num_timesteps = 200  # T
num_trajectories = 10  # N
num_iterations = 250
epochs = 100
#true-true = 16 4e-4
batch_size = 12
learning_rate = 5e-4
eps = 0.2  # clipping



## compute generalised advantage estimates (as done in PPO paper)
def calc_generalized_advantages(rewards, values, gamma=0.99, lambda_=1):
    T = len(rewards)
    advantages = np.zeros(T)  # Initialize the advantages array
    deltas = np.zeros(T)  # To store delta values
    # Compute the deltas (TD residuals)
    for t in range(T - 1): 
        deltas[t] = rewards[t] + gamma * values[t + 1] - values[t]    
    # For the last time step (terminal state), we can assume the value is 0 (or a terminal value if available)
    deltas[T - 1] = rewards[T - 1] - values[T - 1] 
    advantages[T - 1] = deltas[T - 1]  
    for t in range(T - 2, -1, -1): 
        advantages[t] = deltas[t] + gamma * lambda_ * advantages[t + 1]

    return advantages


class PPO:
    def __init__(self, clipping_on, advantage_on, gamma=0.99):

        self.policy_network = PolicyNetwork(3,1)
        self.critic_network = CriticNetwork(3)

        self.optimizer = optim.Adam(list(self.policy_network.parameters()) + list(self.critic_network.parameters()), lr=learning_rate)
    
        self.memory = ReplayBuffer(batch_size)

        self.gamma = gamma
        self.lambda_ = 1
        self.vf_coef = 1  # c1
        self.entropy_coef = 0.01  # c2

        self.clipping_on = clipping_on
        self.advantage_on = advantage_on



    def generate_samples(self):
        current_state = env.reset()
        states = []
        actions = []
        rewards = []
        log_probs = []

        # Run the old policy in the environment for num_timesteps
        for t in range(num_timesteps):
            # Compute mean and std for the current state using the policy network
            mean, std = self.policy_network(torch.as_tensor(current_state))

            # Create a Gaussian distribution and sample an action
            normal = MultivariateNormal(mean, std)
            action = normal.sample().detach()
            log_prob = normal.log_prob(action).detach()

            # Take the sampled action in the environment
            next_state, reward, done, info = env.step(action)

            # Store the results in lists
            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            current_state = next_state

        # Calculate values using the critic network
        values = self.critic_network(torch.as_tensor(states)).squeeze()

        # Calculate advantages
        advantages = calc_generalized_advantages(rewards, values.detach(), self.gamma, self.lambda_)

        # Save the transitions in replay memory
        for t in range(len(rewards)):
            self.memory.push(states[t], actions[t], rewards[t], advantages[t], values[t], log_probs[t])



    def train(self):
        
        train_actor_loss = []
        train_critic_loss = []
        train_total_loss = []
        train_reward = []

        for _ in range(num_iterations): # k

            # collect a number of samples and save the transitions in replay memory
            for _ in range(num_trajectories):
                self.generate_samples()

            # sample from replay memory
            states, actions, rewards, advantages, values, log_probs, batches = self.memory.sample()

            actor_loss_list = []
            critic_loss_list = []
            total_loss_list = []
            reward_list = []
            for _ in range(epochs):

                # calculate the new log prob
                mean,std = self.policy_network(states)
                normal = MultivariateNormal(mean, std)
                new_log_probs = normal.log_prob(actions.unsqueeze(-1))

                r = torch.exp(new_log_probs - log_probs)
                
                if self.clipping_on == True:
                    clipped_r = torch.clamp(r, 1 - eps, 1 + eps)
                else:
                    clipped_r = r

                new_values = self.critic_network(states).squeeze()
                returns = (advantages + values).detach()

                if self.advantage_on == True:
                    actor_loss = (-torch.min(r * advantages, clipped_r * advantages)).mean()
                    critic_loss = nn.MSELoss()(new_values.float(), returns.float())
                else:
                    actor_loss = (-torch.min(r * rewards, clipped_r * rewards)).mean()
                    critic_loss = nn.MSELoss()(new_values.float(), rewards.float())

                # Calcualte total loss
                total_loss = actor_loss + (self.vf_coef * critic_loss) - (self.entropy_coef * normal.entropy().mean())

                # update policy and critic network
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                reward_list.append(sum(rewards))

            # clear replay memory
            self.memory.clear()

            #divide by number of trajectories/episode to get actual loss for an episode
            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_total_loss = sum(total_loss_list) / len(total_loss_list)
            avg_reward = sum(reward_list) / len(reward_list)

            train_actor_loss.append(avg_actor_loss)
            train_critic_loss.append(avg_critic_loss)
            train_total_loss.append(avg_total_loss)
            train_reward.append(avg_reward)

            print('Actor loss = ', avg_actor_loss)
            print('Critic loss = ', avg_critic_loss)
            print('Total Loss = ', avg_total_loss)
            print('Reward = ', avg_reward)
            print("")

        # save the networks
        torch.save(self.policy_network.state_dict(), f'./results/policy_network_{self.clipping_on}_{self.advantage_on}.pt')
        torch.save(self.critic_network.state_dict(), f'./results/critic_network_{self.clipping_on}_{self.advantage_on}.pt')

        fig, axes = plt.subplots(1, 4, figsize=(23, 7))
        axes[0].plot(range(len(train_actor_loss)), train_actor_loss, 'b', label='Actor Loss')
        axes[0].set_title('Actor Loss', fontsize=18)

        axes[1].plot(range(len(train_critic_loss)), train_critic_loss, 'g', label='Critic Loss')
        axes[1].set_title('Critic Loss', fontsize=18)

        axes[2].plot(range(len(train_total_loss)), train_total_loss, 'r', label='Total Loss')
        axes[2].set_title('Total Loss', fontsize=18)

        axes[3].plot(range(len(train_reward)), train_reward, 'orange', label='Accumulated Reward')
        axes[3].set_title('Accumulated Reward', fontsize=18)
        
        fig.suptitle(f'Results for clipping_on={self.clipping_on} and generalized advantage_on={self.advantage_on}\n', fontsize=20)
        fig.tight_layout()
        plt.savefig(f'./results/frame1_{self.clipping_on}_{self.advantage_on}.png')
        fig.show()

        self.show_learning_curve ()
        
    
    def show_learning_curve(self):

        # Sweep theta and theta_dot and find all states
        theta = torch.linspace(-np.pi, np.pi, 100)
        theta_dot = torch.linspace(-8, 8, 100)
        values = torch.zeros((len(theta), len(theta_dot)))

        for i, t in enumerate(theta):
            for j, td in enumerate(theta_dot):
                s = (torch.cos(t), torch.sin(t), td)  
                values[i, j] = self.critic_network(torch.as_tensor(s, dtype=torch.float32))  

        # Display the resulting values using imshow
        fig2 = plt.figure(figsize=(5, 5))
        plt.imshow(values.detach().numpy(), extent=[theta[0], theta[-1], theta_dot[0], theta_dot[-1]], aspect='auto')  # Aspect set to 'auto'
        plt.title('Value grid', fontsize=18)

        plt.xlabel('Angle (theta)', fontsize=18)
        plt.ylabel('Angular velocity (theta_dot)', fontsize=18)

        # Save the figure and display it
        plt.savefig(f'./results/frame2_{self.clipping_on}_{self.advantage_on}.png')
        plt.show()

                
            

    def test(self):

        self.policy_network.load_state_dict(torch.load(f'./results/policy_network_{self.clipping_on}_{self.advantage_on}.pt'))

        s = env.reset()
        
        for i in range(200):

            # compute mu(s) for the current state
            mean,std = self.policy_network(torch.as_tensor(s))

            # the gaussian distribution
            normal = MultivariateNormal(mean, std)

            # sample an action from the gaussian distribution
            action = normal.sample().detach().numpy()

            # emulate taking that action
            s_prime, reward, done, info = env.step(action)

            env.render()

            s = s_prime

        env.close()


 
if __name__ == '__main__':
    user_input = input("Press 0 to run test only.\nPress 1 to run training + test.\n")
    cases = [(True, True), (False, True), (True, False), (False, False)]
    labels = [
        "With Clipping & GAE",
        "Without Clipping & GAE",
        "With Clipping & Without GAE",
        "Without Clipping & Without GAE"
    ]

    print("Select a case:")
    for i, label in enumerate(labels):
        print(f"  {i}: {label}")

    num = int(input("Enter case number: "))
    if num < 0 or num >= len(cases):
        print("Invalid selection!")
    else:
        use_clipping, use_gae = cases[num]
        agent = PPO(use_clipping,use_gae)

        if user_input == '1':
            agent.train()
        agent.test()
