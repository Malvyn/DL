#  coding=utf-8

import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as pyplot


"""
use a random input z to generate a tuple of (x, y) where y = sin(x),
the model we used is VAE.
"""


SAMPLES = 200

class Tensors:
    def __init__(self):
        self.left = tf.placeholder(tf.float32, [None, 2], 'left')
        left = tf.layers.dense(self.left, SAMPLES, activation=tf.nn.relu, name='dense1')
        self.z = tf.layers.dense(left, 1, name='dense2')  # z is the semantics of the input

        z = tf.layers.dense(self.momentum(self.z), SAMPLES, activation=tf.nn.relu, name='dense3')
        self.right = tf.layers.dense(z, 2, name='dense4')

        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.right - self.left), axis=1))
        self.loss = tf.sqrt(loss)
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(loss)

    def momentum(self, z):
        mean = tf.reduce_mean(z, axis=0)
        mean_square = tf.reduce_mean(tf.square(self.z), axis=0)

        self.mom_mean = tf.get_variable('mom_mean', [1], tf.float32, trainable=False, initializer=tf.initializers.zeros)
        self.mom_mean_square = tf.get_variable('mom_mean_square', [1], tf.float32, trainable=False,
                                               initializer=tf.initializers.zeros)
        self.std = tf.sqrt(self.mom_mean_square - self.mom_mean**2)

        self.training = tf.placeholder(tf.bool, [], 'training')

        def training_false():
            return z

        def training_true():
            self.decay = tf.placeholder(tf.float32, [], 'decay')
            new_mom_mean = self.mom_mean * self.decay + mean * (1 - self.decay)
            new_mom_mean_square = self.mom_mean_square * self.decay + mean_square * (1 - self.decay)

            update_mom_mean = tf.assign(self.mom_mean, new_mom_mean)
            update_mom_mean_square = tf.assign(self.mom_mean_square, new_mom_mean_square)
            with tf.control_dependencies([update_mom_mean, update_mom_mean_square]):
                return tf.identity(z)

        return tf.cond(self.training, training_true, training_false)


class Samples:
    def __init__(self):
        self.num = SAMPLES
        start = -math.pi
        end = math.pi

        step = (end - start) / self.num
        self.xs = []

        while start <= end:
            self.xs.append([start, math.sin(start)])
            start += step

    def next_batch(self):
        return self.xs


class GenSin:
    def __init__(self):
        self.samples = Samples()
        self.tensors = Tensors()

    def train(self):
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(5000):
                xs = self.samples.next_batch()
                feed_dict = {
                    self.tensors.left: xs,
                    self.tensors.lr: 0.001,
                    self.tensors.training: True,
                    self.tensors.decay: 0.99
                }

                _, loss, mean, mean_square = session.run([self.tensors.train_op, self.tensors.loss,
                                        self.tensors.mom_mean, self.tensors.mom_mean_square], feed_dict)
                print(loss, mean, mean_square, flush=True)

            num = 200
            mean, std = session.run([self.tensors.mom_mean, self.tensors.std])
            result = []
            for m, s in zip(mean, std):
                result.append(np.random.normal(m, s, [num]))
            result = np.transpose(result, axes=[1, 0])
            feed_dict = {
                self.tensors.z: result,
                self.tensors.training: False
            }
            pairs = session.run(self.tensors.right, feed_dict)

            pairs = sorted(pairs, key=lambda p: p[0])

            pairs = np.transpose(pairs, axes=[1, 0])
            pyplot.plot(pairs[0], pairs[1])
            pyplot.show()


if __name__ == '__main__':
    gs = GenSin()
    gs.train()
