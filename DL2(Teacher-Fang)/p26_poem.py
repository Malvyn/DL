# coding=utf-8


import tensorflow as tf
import numpy as np
from types import MethodType, FunctionType
import argparse


NUM_STEP = (7+1) * 4


class Config:
    def __init__(self):
        self.batch_size = 20
        self.days = 300
        self.num_step = NUM_STEP
        self.lr = 0.0002
        self.epoches = 200
        self.save_path = 'models/p26_poem/poem'
        self.num_units = 200
        self.sample_path = 'data/qts_7X4.txt'
        self.vacabulary_size = 256 * 256
        self.eps = 1e-8
        self.encoding = 'GB18030'

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
                self.x = tf.placeholder(name='x', shape=[config.batch_size, config.num_step], dtype=tf.int32)
                w = tf.get_variable('w', [config.vacabulary_size, config.num_units])
                x = tf.nn.embedding_lookup(w, self.x)  # [batch_size, num_step, num_units]

                cell = tf.nn.rnn_cell.BasicLSTMCell(config.num_units)
                cell = tf.nn.rnn_cell.MultiRNNCell([cell, cell])

                state = cell.zero_state(config.batch_size, tf.float32)
                y_predict = []
                states = []

                loss = 0
                self.y = tf.concat((self.x[:, 1:], tf.zeros([config.batch_size, 1], dtype=tf.int32)), axis=1)
                y = tf.one_hot(self.y, config.vacabulary_size)  # [batch_size, num_step, vacabulary_size]
                for i in range(config.num_step):
                    with tf.variable_scope('multi_lstm_cell'):
                        yi_predict, state = cell(x[:, i, :], state)
                        yi_predict = tf.layers.dense(yi_predict, config.vacabulary_size, name='v2w')
                        #  [batch_size, vacabulary_size]
                        yi_predict = tf.nn.softmax(yi_predict, axis=-1)
                        y_predict.append(tf.argmax(yi_predict, axis=-1, output_type=tf.int32))  # append([batch_size])

                        yi_predict = tf.maximum(config.eps, yi_predict)
                        loss -= tf.reduce_sum(y[:, i, :] * tf.log(yi_predict), axis=-1)
                        states.append(state)
                    tf.get_variable_scope().reuse_variables()
                # y_predict: [num_step, batch_size]
                self.states = states
                self.y_predict = tf.transpose(y_predict, [1, 0])  # [batch_size, num_step]

            with tf.variable_scope('stock_train'):
                self.lr = tf.placeholder(tf.float32, [], 'lr')
                opt = tf.train.AdamOptimizer(self.lr)
                self.loss = tf.reduce_mean(loss / config.num_step)
                self.train_op = opt.minimize(self.loss)
                self.precise = tf.reduce_mean(tf.cast(tf.equal(self.y, self.y_predict), tf.float32))


class Samples:
    def __init__(self, config:Config):
        self.config = config
        file = open(config.sample_path, 'r', encoding='UTF-8')
        lines = file.readlines()

        self.num = len(lines)
        self.data = []
        for i, line in enumerate(lines):
            line = line.rstrip()
            line = line.replace('.', '熵')
            print(i, len(line), line)
            x = self.str2codes(line)
            self.data.append(x)
            print()

        self.point = 0
        self.config = config

    def str2codes(self, line):
        codes = line.encode(self.config.encoding)
        x = []
        for i in range(len(codes) // 2):
            code1, code2 = codes[2 * i], codes[2 * i + 1]
            code = code1 * 256 + code2
            print(code, end=',')
            x.append(code)
        return x

    def next_batch(self, batch_size):
        next_point = self.point + batch_size
        if next_point <= self.num:
            result = self.data[self.point: next_point]
        else:
            result = self.data[self.point:]
            next_point -= self.num
            result = result + self.data[:next_point]
        self.point = next_point
        return result


class Poem:
    def __init__(self, config=Config()):
        graph = tf.Graph()
        with graph.as_default():
            self.config = config
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
            for step in range(self.samples.num // self.config.batch_size):
                xs = self.samples.next_batch(self.config.batch_size)
                feed_dict = {
                    self.tensors.x: xs,
                    self.tensors.lr: self.config.lr
                }

                _, loss, precise = self.session.run([self.tensors.train_op,
                                                     self.tensors.loss, self.tensors.precise], feed_dict)
                print('%d: loss: %.8f, precise: %.8f' % (epoch, loss, precise))
            self.saver.save(self.session, self.config.save_path)

    def predict(self, first_chars):
        xi = self.samples.str2codes(first_chars)
        output = xi.copy()
        x = xi + [0] * (self.config.num_step - len(xi))
        print(x)

        for i in range(self.config.num_step):
            feed_dict = {
                self.tensors.x: [x]
            }
            y = self.session.run(self.tensors.y_predict, feed_dict=feed_dict)
            y = y[0]
            if i < self.config.num_step-1:
                output.append(y[i])
                x[i+len(xi)] = y[i]

        bs = []
        for out in output:
            bs.append(out // 256)
            bs.append(out % 256)
            print(bytes(bs).decode(self.config.encoding), flush=True)
        bs = bytes(bs)
        print(bs.decode(self.config.encoding), flush=True)


if __name__ == '__main__':
    # s = Samples(Config())
    # for _ in range(10):
    #     print(s.next_batch(5))
    #     print('-' * 200)

    # poem = Poem()
    # poem.train()

    config = Config()
    config.batch_size = 1
    poem = Poem(config)
    poem.predict('张')
