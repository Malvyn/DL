# coding=utf-8


import tensorflow as tf
import numpy as np
from types import MethodType, FunctionType
import argparse


class Config:
    def __init__(self):
        self.stocks = 100
        self.days = 300
        self.num_step = (7+1)*4
        self.lr = 0.0001
        self.epoches = 200
        self.save_path = 'models/p26_tf_poem/tf_poem'
        self.num_units = 200
        self.sample_path=''

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
            with tf.variable_scope('stock'):
                self.x = tf.placeholder(name='x', shape=[config.stocks, config.num_step, 4], dtype=tf.float32)
                x = tf.transpose(self.x, [1, 0, 2])
                cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell])
                state = cell.zero_state(config.stocks, tf.float32)
                for i in range(config.num_step):
                    with tf.variable_scope('multi_lstm_cell'):
                        y_predict, state = cell(x[i, :, :], state)
                    tf.get_variable_scope().reuse_variables()

            with tf.variable_scope('stock_train'):
                self.y = tf.placeholder(name='y', shape=[config.stocks, 4], dtype=tf.float32)
                loss = tf.reduce_mean(tf.square(self.y - y_predict))
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                opt = tf.train.AdamOptimizer(self.lr)
                self.train_op = opt.minimize(loss)
                self.loss = tf.sqrt(loss)


class Samples:
    def __init__(self, config:Config):
        self.data = np.random.uniform(0.0, 1.0, [config.stocks, config.days, 4])
        self.num = config.days - config.num_step
        self.point = 0
        self.config = config

    def next_batch(self):
        xs = self.data[:, self.point: self.point + self.config.num_step, :]
        ys = self.data[:, self.point + self.config.num_step, :]
        self.point += 1
        self.point %= self.num
        return xs, ys


class Stock:
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
    stock = Stock()

    stock.train()

