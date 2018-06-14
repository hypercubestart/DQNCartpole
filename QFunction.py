import tensorflow as tf
import random
from collections import deque
import numpy as np

#class to train model
class DQNAgent:
    def __init__(self, sess, state_size, action_size, learning_rate=1e-3, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, gamma=0.95, min_memory=32):
        self.sess = sess #store tensorflow session
        self.epsilon = epsilon #probability take a random action
        self.epsilon_decay = epsilon_decay #rate epsilon decays
        self.epsilon_min = epsilon_min #min value of epsilon
        self.action_size = action_size #different possible actions Ex: 0 = go left, 1 = go right
        self.state_size = state_size
        self.min_memory = min_memory #minimum memory before replay

        self.counter = 0

        self.memory = deque(maxlen=2000)
        self.gamma = gamma #decay of rewards

        self.x = tf.placeholder(tf.float32, shape=[None, state_size], name="x") #state
        self.y = tf.placeholder(tf.float32, shape=[None, action_size], name="y") #expected reward
        with tf.variable_scope("Hidden1"):
            weight = tf.get_variable("weight", shape=[state_size, 24], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("bias", shape=[24], initializer=tf.zeros_initializer)

            hiddenlayer1 = tf.nn.leaky_relu(tf.matmul(self.x, weight) + bias)

        with tf.variable_scope("Hidden2"):
            weight = tf.get_variable("weight", shape=[24, 24], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("bias", shape=[24], initializer=tf.zeros_initializer)

            hiddenlayer2 = tf.nn.leaky_relu(tf.matmul(hiddenlayer1, weight) + bias)

        with tf.variable_scope("Output"):
            weight = tf.get_variable("weight", shape=[24, action_size], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable("bias", shape=[action_size], initializer=tf.zeros_initializer)

            self.output = tf.add(tf.matmul(hiddenlayer2, weight), bias, name="output") #{batch_size, num_actions]


        self.loss = tf.reduce_mean(tf.square(self.y - self.output))

        self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        tf.summary.scalar('learning rate', learning_rate)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('epsilon', self.epsilon)

        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train',
                                             sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if (random.random() < self.epsilon):
            return random.randrange(self.action_size)

        return np.argmax(self.sess.run(self.output, feed_dict={self.x: state})[0])

    def replay(self, batch_size):
        if self.min_memory > len(self.memory):
            return
        minibatch = random.sample(self.memory, batch_size)

        # expected_targets = []
        # states = []

        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.sess.run(self.output, feed_dict={self.x: next_state})) # expected max reward for given action

            # satisfy bellman's equation: reward + gamma * max(Q(next_state)) = Q(state)[action]
            target_f = self.sess.run(self.output, feed_dict={self.x: state}) # get current prediction for state
            target_f[0][action] = target

            summary, _, loss = self.sess.run([self.merged, self.train, self.loss],
                                     feed_dict={self.x: state, self.y: target_f})
            total_loss += loss

            self.train_writer.add_summary(summary, self.counter)
            self.counter += 1

            # states.append(state.flatten())  # state is input value
            # expected_targets.append(target_f.flatten()) # expected target is output
        # summary, _, loss = self.sess.run([self.merged, self.train, self.loss],
        #                                  feed_dict={self.x: states, self.y: expected_targets})
        # self.train_writer.add_summary(summary, self.counter)
        # self.counter+=1


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        print("loss: {}".format(total_loss))

    def save(self, export_dir):
        tf.saved_model.simple_save(self.sess, export_dir=export_dir, inputs={"input": self.x}, outputs={"output": self.output})




