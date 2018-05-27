import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')
env._max_episode_steps = 500

env.reset()

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING], export_dir='./saved_model1960') #1100

    for test in range(10):
        state = env.reset()
        state = np.reshape(state, [1, 4])

        for time_t in range(500):
            env.render()

            action = np.argmax(sess.run("Output/output:0", feed_dict={"x:0": state})[0])

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            state = next_state

            if done:
                # print the score and break out of the loop
                print("test: {}/{}, score: {}"
                      .format(test, 10, time_t))
                break

    env.close()