# coding=utf-8


import tensorflow as tf
import numpy as np
from types import MethodType, FunctionType
import argparse


class Config:
    def __init__(self):
        self.batch_size = 10
        self.num_step1 = 10
        self.num_step2 = 12
        self.lr = 0.001
        self.epoches = 200
        self.save_path = 'models/p24_abstract/abstract'
        self.num_units = 100  # should be 200 at least in a real project
        self.words = 1000  # number of the words
        self.lstm_layers = 2
        self.eps = 1e-5

    def to_dict(self):
        result = {}
        for name in dir(self):
            if name.startswith('__'):
                continue
            attr = getattr(self, name)
            if isinstance(attr, MethodType) or isinstance(attr, FunctionType):
                continue
            print(name, type(attr))
            result[name] = attr
        return result

    def from_cmd_line(self):
        parser = argparse.ArgumentParser()
        attrs = self.to_dict()
        for name in attrs:
            attr = attrs[name]
            parser.add_argument('--' + name, type=type(attr), help='Default to %s' % attr, default=attr)

        a = parser.parse_args()

        for name in dir(a):
            if name in attrs:
                setattr(self, name, getattr(a, name))

    def __repr__(self):
        result = '{'
        attrs = self.to_dict()
        for name in attrs:
            result += '%s: %s; ' % (name, attrs[name])
        return result + '}'


class Tensors:
    def __init__(self, config: Config):
        self.config = config
        with tf.device('/gpu:0'):
            with tf.variable_scope('abstract'):
                self.x = tf.placeholder(name='x', shape=[config.batch_size, config.num_step1], dtype=tf.int32)
                state = self.encode(self.x)
                y_predict = self.decode(state)

            with tf.variable_scope('abstract_train'):
                self.y_predict = [tf.argmax(yi_predict, axis=1) for yi_predict in y_predict]

                self.y = tf.placeholder(name='y', shape=[config.batch_size, config.num_step2], dtype=tf.int32)
                y = tf.one_hot(self.y, config.words)  # [-1, num_step, words]
                y_predict = tf.transpose(y_predict, [1, 0, 2])  # [-1, num_step, words]
                y_predict = tf.maximum(y_predict, config.eps)  # [-1, num_step, words]
                self.loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_predict), axis=2))
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                opt = tf.train.AdamOptimizer(self.lr)
                self.train_op = opt.minimize(self.loss)

    def encode(self, x):
        config = self.config
        vector_dict = tf.get_variable('vector_dict', [config.words, config.num_units], tf.float32)
        x = tf.nn.embedding_lookup(vector_dict, x)  # [-1, num_step, num_units]
        cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units)
        cell_enc = tf.nn.rnn_cell.MultiRNNCell([cell] * config.lstm_layers)
        state = cell_enc.zero_state(config.batch_size, tf.float32)
        with tf.variable_scope('encode'):
            for i in range(config.num_step1):
                _, state = cell_enc(x[:, i, :], state)
                tf.get_variable_scope().reuse_variables()
        return state

    def decode(self, state):
        config = self.config
        x = tf.zeros([config.batch_size, config.num_units])
        cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units)
        cell_dec = tf.nn.rnn_cell.MultiRNNCell([cell] * config.lstm_layers)

        with tf.variable_scope('decode'):
            y_predict = []
            for i in range(config.num_step2):
                hi, state = cell_dec(x, state)  # [-1, num_units]
                logit = tf.layers.dense(hi, config.words, name='dense')  # [-1, words]
                yi_predict = tf.nn.softmax(logit, axis=1)  # [-1, words]
                y_predict.append(yi_predict)
                tf.get_variable_scope().reuse_variables()
                # y_predict.shape = [num_step, -1, words]
            return y_predict


class Samples:
    def __init__(self, config:Config):
        self.num = 100
        self.xs = np.random.randint(0, config.words, [self.num, config.batch_size, config.num_step1])
        self.ys = np.random.randint(0, config.words, [self.num, config.batch_size, config.num_step2])
        self.config = config
        self.point = 0

    def next_batch(self):
        xs = self.xs[self.point, :, :]
        ys = self.ys[self.point, :]
        self.point += 1
        self.point %= self.num
        return xs, ys


class Abstract:
    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.config = Config()
            self.config.from_cmd_line()
            self.tensors = Tensors(self.config)
            self.samples = Samples(self.config)

            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            self.session = tf.Session(graph=graph, config=cfg)
            self.saver = tf.train.Saver()

            try:
                self.saver.restore(self.session, self.config.save_path)
                print('Restore model from %s successfully' % self.config.save_path)
            except:
                print('Fail to restore model from %s' % self.config.save_path)
                self.session.run(tf.global_variables_initializer())

    def train(self):
        for epoch in range(self.config.epoches):
            for step in range(self.samples.num):
                xs, ys = self.samples.next_batch()
                feed_dict = {
                    self.tensors.x: xs,
                    self.tensors.y: ys,
                    self.tensors.lr: self.config.lr
                }

                _, loss = self.session.run([self.tensors.train_op, self.tensors.loss], feed_dict)
                print('%d: loss: %.8f' % (epoch, loss))
            self.saver.save(self.session, self.config.save_path)


if __name__ == '__main__':
    stock = Abstract()

    stock.train()

