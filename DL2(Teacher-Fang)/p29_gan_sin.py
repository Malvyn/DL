# coding=utf-8


import tensorflow as tf
import numpy as np
from types import MethodType, FunctionType
import argparse
import math


"""
Use samples of [x, sin(x)] to train a GAN
"""


class Config:
    def __init__(self):
        self.batch_size = 200
        self.lr = 0.0002
        self.epoches = 200
        self.save_path = 'models/p29_gan_sin/gan_sin'
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
            with tf.variable_scope('gan_sin'):
                self.s = tf.placeholder(tf.float32, [None, 2], 's')
                with tf.variable_scope('generator'):
                    self.z = tf.placeholder(tf.float32, [None, 1], 'z')
                    self.fake = self.generator(self.z)
                with tf.variable_scope('discriminator'):
                    s_p = self.discriminator(self.s)
                    tf.get_variable_scope().reuse_variables()
                    f_p = self.discriminator(self.fake)

            with tf.variable_scope('gan_sin_train'):
                s_p = tf.maximum(config.eps, s_p)
                f_p = tf.maximum(config.eps, f_p)
                self.loss1 = -tf.reduce_mean(tf.log(s_p))
                self.loss2 = -tf.reduce_mean(tf.log(1 - f_p))
                self.loss3 = -tf.reduce_mean(tf.log(f_p))

                self.lr = tf.placeholder(tf.float32, [], 'lr')
                opt = tf.train.AdamOptimizer(self.lr)

                vars = tf.trainable_variables()
                vars_gen = [var for var in vars if 'generator' in var.name]
                vars_dis = [var for var in vars if 'discriminator' in var.name]

                gradients = opt.compute_gradients(self.loss1, vars_dis)
                self.train_op1 = opt.apply_gradients(gradients)

                gradients = opt.compute_gradients(self.loss2, vars_dis)
                self.train_op2 = opt.apply_gradients(gradients)

                gradients = opt.compute_gradients(self.loss3, vars_gen)
                self.train_op3 = opt.apply_gradients(gradients)

    def discriminator(self, x):  #  x.shape: [-1, 2]
        x = tf.layers.dense(x, 200, tf.nn.relu, name='dense1')  # [-1, 200]
        x = tf.layers.dense(x, 1, name='dense2')  # [-1, 1]
        x = tf.nn.sigmoid(x)
        return x  # [-1, 1]

    def generator(self, z):  # z.shape: [-1, 1]
        z = tf.layers.dense(z, 200, tf.nn.relu, name='dense1')  # [-1, 200]
        z = tf.layers.dense(z, 2, name='dense2')  # [-1, 2]
        return z


class Samples:
    def __init__(self, config:Config):
        start = -math.pi
        end = math.pi

        step = (end - start) / config.batch_size
        self.data = []
        for _ in range(config.batch_size):
            self.data.append((start, math.sin(start)))
            start += step

    def next_batch(self):
        return self.data


class GANSin:
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
            data = self.samples.next_batch()
            feed_dict = {
                self.tensors.s: data,
                self.tensors.lr: self.config.lr,
                self.tensors.z: np.random.uniform(size=[self.config.batch_size, 1])
            }
            _, loss1 = self.session.run([self.tensors.train_op1, self.tensors.loss1], feed_dict)
            _, loss2 = self.session.run([self.tensors.train_op2, self.tensors.loss2], feed_dict)
            _, loss3 = self.session.run([self.tensors.train_op3, self.tensors.loss3], feed_dict)
            print('%d, loss1: %.8f; loss2: %.8f; loss3: %.8f' % (epoch, loss1, loss2, loss3))

            self.saver.save(self.session, self.config.save_path)

    def predict(self):
        feed_dict = {
            self.tensors.z: np.random.uniform(size=[self.config.batch_size, 1])
        }
        fake = self.session.run(self.tensors.fake, feed_dict)
        for f in fake:
            print('%.8f, %.8f, %.8f' % (f[0], f[1], math.sin(f[0])))


if __name__ == '__main__':
    import os
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    print(devices)

    # gansin = GANSin()
    #
    # gansin.train()
    # gansin.predict()
