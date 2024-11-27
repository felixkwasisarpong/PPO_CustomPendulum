import gym
from gym import spaces
import numpy as np

class UpdatedPendulumEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def reset(self):
        # Reset environment to initial state
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def step(self, action):
        # Apply action and return (state, reward, done, info)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, False, {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
