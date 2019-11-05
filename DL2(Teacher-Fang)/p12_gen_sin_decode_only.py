#  coding=utf-8

import tensorflow as tf
import math
import matplotlib.pyplot as pyplot


"""
use a random input z to generate a tuple of (x, y) where y = sin(x),
the model we used is VAE.
"""


SAMPLES = 200

class Tensors:
    def __init__(self):
        self.xs = tf.placeholder(tf.float32, [None, 2], 'xs')
        self.index = tf.placeholder(tf.int32, [None], 'left')
        # index = tf.one_hot(self.index, SAMPLES)  # [-1, SAMPLES]
        codebook = tf.get_variable('codebook', [SAMPLES, 1], tf.float32)
        self.z = tf.nn.embedding_lookup(codebook, self.index)

        z = tf.layers.dense(self.z, SAMPLES, activation='relu', name='dense3')
        self.right = tf.layers.dense(z, 2, name='dense4')

        loss = tf.reduce_mean(tf.square(self.right - self.xs))
        self.loss = tf.sqrt(loss)
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(loss)


class Samples:
    def __init__(self):
        self.num = SAMPLES
        start = -math.pi
        end = math.pi

        step = (end - start) / self.num
        self.xs = []

        for _ in range(self.num):
            self.xs.append([start, math.sin(start)])
            start += step

        self.index = [e for e in range(len(self.xs))]

    def next_batch(self):
        return self.index, self.xs


class GenSin:
    def __init__(self):
        self.samples = Samples()
        self.tensors = Tensors()

    def train(self):
        index, xs = self.samples.next_batch()
        feed_dict = {
            self.tensors.index: index,
            self.tensors.xs: xs,
            self.tensors.lr: 0.0005
        }
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(5000):
                _, loss = session.run([self.tensors.train_op, self.tensors.loss], feed_dict)
                print(loss, flush=True)


if __name__ == '__main__':
    gs = GenSin()
    gs.train()
