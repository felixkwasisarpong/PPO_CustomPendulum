

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
from Modules import CriticNetwork,PolicyNetwork,ReplayBuffer,RecurrentPolicyNetwork
from torch.distributions import MultivariateNormal,Normal

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




num_critics = 5

hidden_dim = 64
# Sample hyperparameters
num_timesteps = 200 # T
num_trajectories = 10 # N
num_iterations = 250
epochs = 100
episodes = 200
batch_size = 10
learning_rate = 1e-4
eps = 0.2 # clipping


## compute generalised advantage estimates (as done in PPO paper)
def calc_generalized_advantages(rewards, values, gamma=0.99, lambda_=1):
    advantages = torch.zeros_like(torch.as_tensor(rewards))
    sum = 0
    for t in reversed(range(len(rewards)-1)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        sum = delta + gamma * lambda_ * sum
        advantages[t] = sum
    
    return advantages



class PPO:
    def __init__(self, clipping_on, advantage_on, gamma=0.99):
        self.policy_network = RecurrentPolicyNetwork(input_size=2)
        self.critics_network = torch.nn.ModuleList([CriticNetwork(input_size=2) for _ in range(num_critics)])


        self.optimizer = torch.optim.Adam(
            [
                {'params': self.policy_network.parameters(), 'lr': learning_rate},
                {'params': [p for critic in self.critics_network for p in critic.parameters()], 'lr': learning_rate}
            ]
        )


        self.memory = ReplayBuffer(batch_size)
        self.gamma = gamma
        self.lambda_ = 1
        self.vf_coef = 1
        self.entropy_coef = 0.01
        self.clipping_on = clipping_on
        self.advantage_on = advantage_on
        self.std = torch.diag(torch.full(size=(1,), fill_value=0.5))
        
    def compute_values(self, states):
        states_tensor = torch.as_tensor(states)
        critic_values = [critic(states_tensor).squeeze() for critic in self.critics_network]
        values = torch.mean(torch.stack(critic_values), dim=0)
        return values

    
    def generate_samples(self):
        
        obs = env.reset()
        current_state = torch.tensor([obs[:2]])

        states = []
        actions = []
        rewards = []
        log_probs = []
        

        # Run the old policy in environment for num_timestep            
        for t in range(num_timesteps):
            
            # compute mu(s) for the current state
            mean = self.policy_network(current_state)

            # the gaussian distribution
            normal = MultivariateNormal(mean, self.std)

            # sample an action from the gaussian distribution
            action = normal.sample().detach()
            log_prob = normal.log_prob(action).detach()

            states.append(obs.flatten()[:2])

            # emulate taking that action
            obs, reward, done, info = env.step(action.tolist())
            next_state = torch.tensor([obs.flatten()[:2]])

            # store results in a list
            actions.append(action)
            rewards.append(torch.as_tensor(reward))
            log_probs.append(log_prob)
            
            #env.render()

            current_state = next_state
        # calculate values
        values = self.compute_values(states)

        advantages = calc_generalized_advantages(rewards, values.detach(), self.gamma, self.lambda_)

        for t in range(len(rewards)):
            self.memory.push(states[t], actions[t], rewards[t], advantages[t], values[t], log_probs[t])

    def train(self):
        train_actor_loss, train_critic_loss, train_total_loss, train_reward = [], [], [], []

        for _ in range(num_iterations): 
            for _ in range(num_trajectories):
                self.generate_samples()

            states, actions, rewards, advantages, values, log_probs, _ = self.memory.sample()

            actor_loss_list, critic_loss_list, total_loss_list, reward_list = [], [], [], []

            for _ in range(epochs):
                hidden = (torch.zeros(1, states.size(0), 64), torch.zeros(1, states.size(0), 64))
                mean = self.policy_network(states)
                normal = MultivariateNormal(mean, self.std)
                new_log_probs = normal.log_prob(actions.unsqueeze(-1))
                
                r = torch.exp(new_log_probs - log_probs)
                clipped_r = torch.clamp(r, 1 - eps, 1 + eps) if self.clipping_on else r
                
                new_values = self.compute_values(states)
                returns = (advantages + values).detach()

                if self.advantage_on:
                    actor_loss = (-torch.min(r * advantages, clipped_r * advantages)).mean()
                else:
                    actor_loss = (-torch.min(r * rewards, clipped_r * rewards)).mean()

                critic_loss = nn.MSELoss()(new_values.float(), returns.float())
                total_loss = actor_loss + self.vf_coef * critic_loss - self.entropy_coef * normal.entropy().mean()

                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                self.optimizer.step()

                actor_loss_list.append(actor_loss.item())
                critic_loss_list.append(critic_loss.item())
                total_loss_list.append(total_loss.item())
                reward_list.append(sum(rewards))

            self.memory.clear()

            avg_actor_loss = sum(actor_loss_list) / len(actor_loss_list)
            avg_critic_loss = sum(critic_loss_list) / len(critic_loss_list)
            avg_total_loss = sum(total_loss_list) / len(total_loss_list)
            avg_reward = sum(reward_list) / len(reward_list)

            train_actor_loss.append(avg_actor_loss)
            train_critic_loss.append(avg_critic_loss)
            train_total_loss.append(avg_total_loss)
            train_reward.append(avg_reward)

            print(f'Actor loss = {avg_actor_loss}')
            print(f'Critic loss = {avg_critic_loss}')
            print(f'Total Loss = {avg_total_loss}')
            print(f'Reward = {avg_reward}\n')

        # save the networks
        torch.save(self.policy_network.state_dict(), f'./results/policy_network_{self.clipping_on}_{self.advantage_on}.pt')
        torch.save(self.critics_network.state_dict(), f'./results/critic_network_{self.clipping_on}_{self.advantage_on}.pt')

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
                s = (torch.cos(t), td)  
                
                # Average the output of the ensemble critics
                critic_values = torch.zeros(len(self.critics_network))  # Assuming self.critic_networks is a list or module list
                
                for k, critic in enumerate(self.critics_network):
                    critic_values[k] = critic(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))  # Get value from each critic
                
                # Average over all critics
                values[i, j] = critic_values.mean()

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

        obs = env.reset()
        current_state = obs[:2]

        angle_list = []
        
        for i in range(episodes):

            # compute mu(s) for the current state
            mean = self.policy_network(torch.as_tensor(current_state).squeeze().unsqueeze(0))

            # the gaussian distribution
            normal = MultivariateNormal(mean, self.std)

            # sample an action from the gaussian distribution
            action = normal.sample().detach().numpy()

            # save the state in a list
            angle_list.append(np.arccos(current_state[0].item()))

            # emulate taking that action
            obs, reward, done, info = env.step(action.tolist())
            next_state = obs[:2]

            env.render(mode="human")

            current_state = next_state

        env.close()


        fig = plt.figure(figsize=(5, 5))
        plt.plot(range(len(angle_list)), angle_list, 'r')
        plt.title('Angle VS Time', fontsize=18)
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Angle', fontsize=18)
        
        plt.savefig(f'./results/figure2.png')
        fig.show()



 
if __name__ == '__main__':#
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
