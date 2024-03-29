import numpy as np
import matplotlib
from collections import deque

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_LEFT_DOWN = 4
ACTION_LEFT_UP = 5
ACTION_RIGHT_DOWN = 6
ACTION_RIGHT_UP = 7

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# discount for max value
GAMMA = 1.0

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT,
           ACTION_LEFT_DOWN, ACTION_LEFT_UP, ACTION_RIGHT_DOWN, ACTION_RIGHT_UP]


# viento aleatorio: np.random.choice(np.arange(WIND[j] + 1))
def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - np.random.choice(np.arange(WIND[j] + 1)), 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - np.random.choice(np.arange(WIND[j] + 1)), WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - np.random.choice(np.arange(WIND[j] + 1)), 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - np.random.choice(np.arange(WIND[j] + 1)), 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_LEFT_DOWN:
        return [max(min(i + 1 - np.random.choice(np.arange(WIND[j] + 1)), WORLD_HEIGHT - 1), 0), max(j - 1, 0)]
    elif action == ACTION_LEFT_UP:
        return [max(i - 1 - np.random.choice(np.arange(WIND[j] + 1)), 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT_DOWN:
        return [max(min(i + 1 - np.random.choice(np.arange(WIND[j] + 1)), WORLD_HEIGHT - 1), 0),
                min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_RIGHT_UP:
        return [max(i - 1 - np.random.choice(np.arange(WIND[j] + 1)), 0), min(j + 1, WORLD_WIDTH - 1)]
    else:
        assert False


# play for an episode
def episode(q_value, sarsa=True):
    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        if sarsa:
            # Sarsa update
            q_value[state[0], state[1], action] += ALPHA * (
                    REWARD + q_value[next_state[0], next_state[1], next_action] - q_value[state[0], state[1], action])
        else:
            # Q-Learning update
            q_value[state[0], state[1], action] += ALPHA * (
                    REWARD + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) - q_value[
                state[0], state[1], action])

        state = next_state
        action = next_action
        time += 1
    return time


def activity_1_3():
    results = []

    for e in range(2):
        avg = deque(maxlen=20)
        q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 8))
        episode_limit = 500

        steps = []
        ep = 0
        while ep < episode_limit:
            current_step = episode(q_value, e == 1)
            avg.append(current_step)
            steps.append(np.round(np.mean(avg)))
            ep += 1

        show_policy(q_value)
        results.append(steps)

    plt.plot(np.arange(1, len(steps) + 1), results[0], color='red')
    plt.plot(np.arange(1, len(steps) + 1), results[1], color='blue')
    plt.legend(['Q-Learning', 'Sarsa(0)'])
    plt.xlabel('Episodes')
    plt.ylabel('Time steps')
    plt.show()


def show_policy(q_value):
    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G ')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U ')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D ')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L ')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R ')
            elif bestAction == ACTION_LEFT_UP:
                optimal_policy[-1].append('LU')
            elif bestAction == ACTION_LEFT_DOWN:
                optimal_policy[-1].append('LD')
            elif bestAction == ACTION_RIGHT_UP:
                optimal_policy[-1].append('RU')
            elif bestAction == ACTION_RIGHT_DOWN:
                optimal_policy[-1].append('RD')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) + ' ' for w in WIND]))


activity_1_3()
