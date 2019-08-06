import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from GPyOpt.methods import BayesianOptimization


class QLearningSolver:
    def __init__(self, name, n_episodes=1000, alpha=0.1, epsilon=0.1, gamma=1.0):
        self.buckets = (1, 1, 6, 12,)  # down-scaling feature space to discrete range
        self.n_episodes = n_episodes  # training episodes
        self.n_win_ticks = 200  # average ticks over 100 episodes required for win
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.env = gym.make(name)
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (
                reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())

            done = False
            i = 0

            while not done:
                # self.env.render()
                action = self.choose_action(current_state, self.epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state, self.alpha)
                current_state = new_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return mean_score
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        print('Did not solve after {} episodes 😞'.format(e))
        return mean_score


def activity_2_1():

    def objective(params):
        print(params)
        solver = QLearningSolver('CartPole-v1', alpha=params[0][0], epsilon=params[0][1])
        res = solver.run()
        print(res)
        return res

    bds = [
        {'name': 'alpha', 'type': 'continuous', 'domain': (0, 1)},
        {'name': 'epsilon', 'type': 'continuous', 'domain': (0, 1)}
    ]

    # define el optimizador
    optimizer = BayesianOptimization(f=objective,
                                     domain=bds,
                                     model_type='GP',
                                     acquisition_type='EI',
                                     acquisition_jitter=0.05,
                                     verbosity=True,
                                     maximize=True)

    # realiza las 20 iteraciones de la optimizacion
    optimizer.run_optimization(max_iter=5)

    print(optimizer.Y)

    plt.contour(optimizer.X[:, 0], optimizer.X[:, 0], optimizer.Y)


activity_2_1()