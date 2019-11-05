# coding=utf-8


import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import cv2

SAMPLE_PATH = 'MNIST_data'
SAVE_PATH = 'models/mnist02/mnist'


class Tensors:
    def __init__(self):
        x = tf.placeholder(tf.float32, [None, 28*28], name='x')
        self.x = x

        x = tf.reshape(x, [-1, 28, 28, 1])
        # x = tf.layers.conv2d(x, 64, 3, 2, 'same', activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 32, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)  # ==> 14 * 14 * 32

        x = tf.layers.conv2d(x, 64, 3, 1, 'same', activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)  # ==> 7 * 7 * 64
        x = tf.layers.flatten(x)

        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        x = tf.nn.dropout(x, self.keep_prob)

        x = tf.layers.dense(x, 1000, activation=tf.nn.relu)

        x = tf.layers.dense(x, 10)
        x = tf.nn.softmax(x)
        self.y_predict = tf.argmax(x, axis=1)

        y = tf.placeholder(tf.int32, [None], name='y')
        self.y = y
        y = tf.one_hot(y, 10)

        x = tf.maximum(x, 1e-10)
        self.loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(x), axis=-1))
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss)


class MNIST:
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.ts = Tensors()
            self.session = tf.Session(graph=graph)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.session, SAVE_PATH)
                print('Success to restore the model from %s' % SAVE_PATH, flush=True)
            except:
                print('Fail to restore the mode from %s, use a new model instead.' % SAVE_PATH, flush=True)
                self.session.run(tf.global_variables_initializer())

    def train(self, batch_size=10, epoches=100, lr=0.0001, keep_prob=0.5):
        ds = input_data.read_data_sets(SAMPLE_PATH)
        samples = ds.train.num_examples
        steps = samples // batch_size

        for epoch in range(epoches):
            for step in range(steps):
                xs, ys = ds.train.next_batch(batch_size)
                feed_dict = {
                    self.ts.x: xs,
                    self.ts.y: ys,
                    self.ts.lr: lr,
                    self.ts.keep_prob: keep_prob
                }
                _, loss = self.session.run([self.ts.train_op, self.ts.loss], feed_dict)

                print('epoch: %d, step: %d, loss: %.8f' % (epoch, step, loss), flush=True)
            self.saver.save(self.session, SAVE_PATH)
            print('Model is saved into %s' % SAVE_PATH)


def show_sample():
    ds = input_data.read_data_sets(SAMPLE_PATH)
    xs, ys = ds.train.next_batch(20)

    xs = np.reshape(xs, [-1, 28, 28])
    xs = xs * 255

    for x in xs:
        cv2.imshow('mnist', x)
        cv2.waitKey(3000)


if __name__ == '__main__':
    # show_sample()

    mnist = MNIST()
    mnist.train()
