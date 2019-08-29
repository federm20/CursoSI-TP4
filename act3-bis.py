import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from GPyOpt.methods import BayesianOptimization


class MountainCarSolver:
    def __init__(self, n_episodes=1000, alpha=0.1, epsilon=0.1, gamma=1.0):
        self.buckets = (18, 15,)
        self.n_episodes = n_episodes  # training episodes
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.env = gym.make('MountainCarContinuous-v0')
        self.q_value = np.zeros((18, 15, 2, 1))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)


    def discretize_action(self, action):
        return [0] if action < 0 else [1]


    def hedger_prediction(query_point, h):
        self.


    def hedger_trainig():



def activity_3(double=False):
    solver = MountainCarSolver(double=double, alpha=0.2, epsilon=0.1)
    return solver.hedger_trainig()


activity_3()
