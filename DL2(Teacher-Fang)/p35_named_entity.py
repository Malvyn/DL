# coding=utf-8


import tensorflow as tf
import numpy as np
from types import MethodType, FunctionType
import argparse


class Config:
    def __init__(self):
        self.batch_size = 100
        self.num_step = 20
        self.lr = 0.0001
        self.epoches = 200
        self.save_path = 'models/p35_named_entity/named_entity'
        self.num_units = 4
        self.eps = 1e-8

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
            with tf.variable_scope('named_entity'):
                self.x = tf.placeholder(name='x', shape=[config.batch_size, config.num_step, config.num_units], dtype=tf.float32)
                x = tf.transpose(self.x, [1, 0, 2])

                self.y = tf.placeholder(name='y', shape=[config.batch_size, config.num_step], dtype=tf.int32)
                y = tf.transpose(self.y, [1, 0])
                y = tf.one_hot(y, 10)  # [num_step, batch_size, 10]

                cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units)
                # cell = tf.nn.rnn_cell.DropoutWrapper(cell, keep_prob)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
                cell = MyBidirectionCell(cell, cell)

                state = cell.zero_state(config.batch_size)

                with tf.variable_scope('binary_lstm_cell'):
                    y_predict, _ = cell(x, state)  # [num_step, batch_size, num_units]
                    y_predict = tf.convert_to_tensor(y_predict)
                    y_predict = tf.layers.dense(y_predict, 10, name='y_dense1')
                    y_predict = tf.nn.softmax(y_predict)  # [num_step, batch_size, 10]

            with tf.variable_scope('named_entity_train'):
                self.loss = -tf.reduce_mean(y * tf.log(tf.maximum(config.eps, y_predict)))
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                opt = tf.train.AdamOptimizer(self.lr)
                self.train_op = opt.minimize(self.loss)


class MyBidirectionCell:
    def __init__(self, cell_pos, cell_neg):
        self.cell_pos = cell_pos
        self.cell_neg = cell_neg

    def __call__(self, x, state):
        state_pos, state_neg = state
        num_units = x[0].shape[1].value
        yi_pos_s = []
        yi_neg_s = []
        with tf.variable_scope('for_1'):
            for i in range(x.shape[0].value):
                xi_pos = x[i, :, :]
                xi_neg = x[-(i+1), :, :]
                yi_pos, state_pos = self.cell_pos(xi_pos, state_pos)
                yi_neg, state_neg = self.cell_neg(xi_neg, state_neg)
                yi_pos_s.append(yi_pos)
                yi_neg_s.append(yi_neg)
                tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('for_2'):
            y = []
            for i, yi_pos, yi_neg in zip(range(len(yi_pos_s)), yi_pos_s, reversed(yi_neg_s)):
                yi = tf.concat((yi_pos, yi_neg), axis=1)
                yi = tf.layers.dense(yi, num_units, name='dense%d' % i)
                y.append(yi)
        return y, (state_pos, state_neg)

    def zero_state(self, batch_size):
        return self.cell_pos.zero_state(batch_size, dtype=tf.float32), \
               self.cell_neg.zero_state(batch_size, dtype=tf.float32)


class Samples:
    def __init__(self, config:Config):
        self.num = 100
        self.xs = np.random.uniform(0.0, 1.0, [self.num, config.batch_size, config.num_step, config.num_units])
        self.ys = np.random.randint(0, 10, [self.num, config.batch_size, config.num_step], np.int32)
        self.point = 0
        self.config = config

    def next_batch(self):
        x = self.xs[self.point, :, :, :]
        y = self.ys[self.point, :, :]
        self.point += 1
        self.point %= self.num
        return x, y


class NamedEntity:
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
                x, y = self.samples.next_batch()
                feed_dict = {
                    self.tensors.x: x,
                    self.tensors.y: y,
                    self.tensors.lr: self.config.lr
                }

                _, loss = self.session.run([self.tensors.train_op, self.tensors.loss], feed_dict)
                print('%d: loss: %.8f' % (epoch, loss))
            self.saver.save(self.session, self.config.save_path)


if __name__ == '__main__':
    # Tensors(Config())
    stock = NamedEntity()
    stock.train()

