import gym
import numpy as np
import math

buckets = (1, 1, 6, 12,)

def discretize(env, obs):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

def activity_2_1():
    env = gym.make("CartPole-v1")
    observation = env.reset()

    # Q = np.zeros((env.observation_space.shape, env.action_space.n))
    # print(Q)
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        print(discretize(env, observation))

        # TODO: Q learning

        print(observation, reward, done, info)

        # if done:
        #   observation = env.reset()
    env.close()

activity_2_1()
