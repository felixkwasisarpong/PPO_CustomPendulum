import gym
import a3_gym_env
env = gym.make('Pendulum-v1-custom')

# sample hyperparameters
batch_size = 10000
epochs = 30
learning_rate = 1e-2
hidden_size = 8
n_layers = 2
