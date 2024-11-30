# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1afVchWmS1JUj9HSytGtQFb8EPu_ri6bz
"""

def train():
    env = gym.make('Pendulum-v1')
    agent = PPOAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    num_episodes = 1000
    max_timesteps = 200
    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, log_prob, value)
            episode_reward += reward

            if done or t == max_timesteps - 1:
                agent.update(next_state)
                break

            state = next_state

        rewards_history.append(episode_reward)
        print(f"Episode {episode}, Reward: {episode_reward}")

    # Plotting learning curve
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Learning Curve')
    plt.show()

train()