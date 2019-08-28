import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
from GPyOpt.methods import BayesianOptimization


class MountainCarSolver:
    def __init__(self, n_episodes=1000, alpha=0.1, epsilon=0.1, gamma=1.0, double=False):
        self.buckets = (18, 15,)
        self.n_episodes = n_episodes  # training episodes
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.gamma = gamma  # discount factor
        self.env = gym.make('MountainCarContinuous-v0')
        self.Q1 = np.zeros((18, 15, 2, 1))
        self.double = double

        # double Q-Learning
        if self.double:
            self.Q2 = np.zeros((18, 15, 2, 1))

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def discretize_action(self, action):
        return [0] if action < 0 else [1]

    def choose_action(self, state, epsilon):
        if not self.double:
            return self.discretize_action(self.env.action_space.sample()) if (np.random.random() <= epsilon) \
                else [np.argmax(self.Q1[state])]
        else:
            return self.discretize_action(self.env.action_space.sample()) if (np.random.random() <= epsilon) \
                else [np.argmax([item1 + item2 for item1, item2 in zip(self.Q1[state], self.Q2[state])])]

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
        total_rewards = []
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())

            done = False
            sum_reward = 0

            step = 0

            while not done:
                # self.env.render()
                action = self.choose_action(current_state, self.epsilon)
                obs, reward, done, _ = self.env.step([-1] if action[0] == 0 else [1])
                sum_reward += reward
                step += 1
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state, self.alpha)
                current_state = new_state

            # if sum_reward > 90:
            #     return sum_reward

            # total_rewards.append(max(sum_reward, -1000 if len(total_rewards) == 0 else np.max(total_rewards)))
            scores.append(step)
            total_rewards.append(custom_filter(scores))

        # plt.plot(range(self.n_episodes), total_rewards)
        # plt.show()

        return total_rewards


def custom_filter(array):
    if len(array) > 2:
        return np.mean(np.sort(array)[:50])
    else:
        return np.mean(array)


def activity_2_3_a(double=False):
    def objective(params):
        solver = MountainCarSolver(double=double, alpha=params[0][0], epsilon=params[0][1])
        return solver.run()

    # descomentar para optimizar
    # bayesian_optimization(objective)

    # Descomentar posterior a la optimizacion bayesiana, para obtener resultados del metodo q-learning
    objective([[0.15, 0.15]])


def activity_2_3_b():
    activity_2_3_a(double=True)


def bayesian_optimization(objective):

    bds = [
        {'name': 'alpha', 'type': 'discrete', 'domain': np.arange(0.05, 0.4, 0.05)},
        {'name': 'epsilon', 'type': 'discrete', 'domain': np.arange(0.05, 0.4, 0.05)}
        # {'name': 'gamma', 'type': 'discrete', 'domain': np.arange(0.5, 1, 0.05)}
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
    optimizer.run_optimization(max_iter=30)

    print(optimizer.X)
    print(optimizer.Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = optimizer.X[:, 0].reshape(len(optimizer.X[:, 0]), 1).reshape(-1)
    yy = optimizer.X[:, 1].reshape(len(optimizer.X[:, 1]), 1).reshape(-1)
    zz = -optimizer.Y.reshape(-1)

    surf = ax.plot_trisurf(xx, yy, zz, cmap='viridis')
    fig.colorbar(surf)
    plt.xlabel('Alpha')
    plt.ylabel('Epsilon')
    plt.title('MountainCarContinuos V0 Optimization')
    plt.show()


def compare_activity():
    def objective(params, double):
        solver = MountainCarSolver(double=double, alpha=params[0][0], epsilon=params[0][1])
        return solver.run()

    total_q1 = objective([[0.15, 0.15]], False)
    total_q2 = objective([[0.15, 0.15]], True)

    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.title('Mountain Car Continuos')
    plt.plot(range(1000), total_q1, color='blue')
    plt.plot(range(1000), total_q2, color='red')
    plt.legend(['Q-Learning', 'Double Q-Learning'])
    plt.show()


# activity_2_3_a()
# activity_2_3_b()
compare_activity()
