import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from GPyOpt.methods import BayesianOptimization


class MountainCarSolver:
    def __init__(self, n_episodes=1000, alpha=0.1, epsilon=0.1, gamma=1.0, double=False):
        self.n_episodes = n_episodes  # training episodes
        self.n_win_ticks = 190  # average ticks over 100 episodes required for win
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.env = gym.make('CartPole-v1')
        # self.Q1 = np.zeros(self.buckets + (self.env.action_space.n,))
        self.double = double

        # double Q-Learning
        # if self.double:
        #     self.Q2 = np.zeros(self.buckets + (self.env.action_space.n,))

    def discretize(self, obs):
        state_adj = (obs - self.env.observation_space.low)*np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)
        print(state_adj)
        return state_adj

    def choose_action(self, state, epsilon):
        if not self.double:
            return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q1[state])
        else:
            return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
                [item1 + item2 for item1, item2 in zip(self.Q1[state], self.Q2[state])])

    def update_q(self, state_old, action, reward, state_new, alpha):
        if not self.double:
            self.Q1[state_old][action] += alpha * (
                    reward + self.gamma * np.max(self.Q1[state_new]) - self.Q1[state_old][action])
        else:
            if np.random.binomial(1, 0.5) == 1:
                active_q = self.Q1
                target_q = self.Q2
            else:
                active_q = self.Q2
                target_q = self.Q1

            best_action = np.random.choice([action_ for action_, value_ in enumerate(active_q[state_new]) if
                                            value_ == np.max(active_q[state_new])])
            active_q[state_old][action] += alpha * (
                    reward + self.gamma * target_q[state_new][best_action] - active_q[state_old][action])

    def run(self):
        scores = deque(maxlen=100)
        # total_scores = []

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
            # total_scores.append(mean_score)
            if mean_score >= self.n_win_ticks and e >= 100:
                print('Ran {} episodes. Solved after {} trials ✔'.format(e, e - 100))
                return mean_score
            if e % 100 == 0:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        print('Did not solve after {} episodes 😞'.format(e))

        # plt.plot(range(1000), total_scores)
        # plt.show()

        return mean_score


def activity_2_3_a(double=False):
    def objective(params):
        solver = MountainCarSolver(double=double, alpha=params[0][0], epsilon=params[0][1])
        return solver.run()

    objective([[0.3, 0.275]])

    # bds = [
    #     {'name': 'alpha', 'type': 'discrete', 'domain': np.arange(0.05, 0.4, 0.05)},
    #     {'name': 'epsilon', 'type': 'discrete', 'domain': np.arange(0.05, 0.4, 0.05)}
    #     # {'name': 'gamma', 'type': 'discrete', 'domain': np.arange(0.5, 1, 0.05)}
    # ]
    #
    # # define el optimizador
    # optimizer = BayesianOptimization(f=objective,
    #                                  domain=bds,
    #                                  model_type='GP',
    #                                  acquisition_type='EI',
    #                                  acquisition_jitter=0.05,
    #                                  verbosity=True,
    #                                  maximize=True)
    #
    # # realiza las 20 iteraciones de la optimizacion
    # optimizer.run_optimization(max_iter=30)
    #
    # print(optimizer.X)
    # print(optimizer.Y)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # xx = optimizer.X[:, 0].reshape(len(optimizer.X[:, 0]), 1).reshape(-1)
    # yy = optimizer.X[:, 1].reshape(len(optimizer.X[:, 1]), 1).reshape(-1)
    # zz = -optimizer.Y.reshape(-1)
    #
    # surf = ax.plot_trisurf(xx, yy, zz, cmap='viridis')
    # fig.colorbar(surf)
    # plt.xlabel('Alpha')
    # plt.ylabel('Epsilon')
    # plt.title('Cart Pole V1 Optimization')
    # plt.show()



def activity_2_3_b():
    activity_2_3_a(double=True)


activity_2_3_a()
# activity_2_3_b()
