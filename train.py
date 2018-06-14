##FILE TO TRAIN MODEL
import gym
from QFunction import DQNAgent
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env._max_episode_steps = 500

env.reset()

num_episodes = 3001

move_left_times = 0
move_right_times = 0

with tf.Session() as sess:

    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = DQNAgent(sess, state_space, action_space)

    ###running_total of score
    total_score = 0

    for e in range(num_episodes): #number of games to play
        state = env.reset()
        state = np.reshape(state, [1, 4])

        for time_t in range(500):
            #env.render()

            action = agent.act(state)

            if action == 0:
                move_left_times+=1
            else:
                move_right_times+=1

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            if done:
                reward = -10

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                total_score += time_t
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, num_episodes, time_t))
                break

        agent.replay(32)

        if e % 10 == 0:
            if total_score >= 4500:
                agent.save('./saved_model' + str(e))
            total_score = 0

    plt.bar([0, 1], [move_left_times, move_right_times]) #plot graph
    plt.show()

