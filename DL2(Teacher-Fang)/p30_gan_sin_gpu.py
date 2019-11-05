# coding=utf-8


import tensorflow as tf
import numpy as np
from types import MethodType, FunctionType
import argparse
import math
import os

"""
Use samples of [x, sin(x)] to train a GAN
"""


class Config:
    def __init__(self):
        self.batch_size = 200
        self.lr = 0.0002
        self.epoches = 200
        self.save_path = 'models/p30_gan_sin_gpu/gan_sin_gpu'
        self.eps = 1e-5
        self.gpus = get_gpus()

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
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        opt = tf.train.AdamOptimizer(self.lr)

        self.s_s = []
        self.z_s = []
        self.fake_s = []
        gradients1_s = []
        gradients2_s = []
        gradients3_s = []
        loss1_s = []
        loss2_s = []
        loss3_s = []
        with tf.variable_scope('gan_sin_gpu'):
            for i in range(config.gpus):   # target function, loss function, gradients should be here
                with tf.device('/gpu:%d' % i):
                    s = tf.placeholder(tf.float32, [None, 2], 's')
                    self.s_s.append(s)
                    with tf.variable_scope('generator'):
                        z = tf.placeholder(tf.float32, [None, 1], 'z')
                        self.z_s.append(z)
                        fake = self.generator(z)
                        self.fake_s.append(fake)
                    with tf.variable_scope('discriminator'):
                        s_p = self.discriminator(s)
                        tf.get_variable_scope().reuse_variables()
                        f_p = self.discriminator(fake)
                    with tf.variable_scope('loss_gradient'):
                        s_p = tf.maximum(config.eps, s_p)
                        f_p = tf.maximum(config.eps, f_p)
                        loss1 = -tf.reduce_mean(tf.log(s_p))
                        loss2 = -tf.reduce_mean(tf.log(1 - f_p))
                        loss3 = -tf.reduce_mean(tf.log(f_p))
                        loss1_s.append(loss1)
                        loss2_s.append(loss2)
                        loss3_s.append(loss3)

                        vars = tf.trainable_variables()
                        vars_gen = [var for var in vars if 'generator' in var.name]
                        vars_dis = [var for var in vars if 'discriminator' in var.name]

                        gradients1_s.append(opt.compute_gradients(loss1, vars_dis))
                        gradients2_s.append(opt.compute_gradients(loss2, vars_dis))
                        gradients3_s.append(opt.compute_gradients(loss3, vars_gen))

                tf.get_variable_scope().reuse_variables()

        with tf.variable_scope('gan_sin_train'):
            with tf.device('/gpu:0'):
                self.train_op1 = opt.apply_gradients(avg_gradients(gradients1_s))
                self.train_op2 = opt.apply_gradients(avg_gradients(gradients2_s))
                self.train_op3 = opt.apply_gradients(avg_gradients(gradients3_s))

        self.loss1 = tf.reduce_mean(loss1_s)
        self.loss2 = tf.reduce_mean(loss2_s)
        self.loss3 = tf.reduce_mean(loss3_s)

    def discriminator(self, x):  #  x.shape: [-1, 2]
        x = tf.layers.dense(x, 200, tf.nn.relu, name='dense1')  # [-1, 200]
        x = tf.layers.dense(x, 1, name='dense2')  # [-1, 1]
        x = tf.nn.sigmoid(x)
        return x  # [-1, 1]

    def generator(self, z):  # z.shape: [-1, 1]
        z = tf.layers.dense(z, 200, tf.nn.relu, name='dense1')  # [-1, 200]
        z = tf.layers.dense(z, 2, name='dense2')  # [-1, 2]
        return z


def avg_gradients(gradients_s):
    result = {}
    for gs in gradients_s:
        for grad, var in gs:
            if not var in result:
                result[var] = []
            result[var].append(grad)
    result = [(tf.reduce_mean(result[var], axis=0), var) for var in result]
    return result


class Samples:
    def __init__(self, config:Config):
        self.config = config
        start = -math.pi
        end = math.pi

        step = (end - start) / config.batch_size
        self.data = []
        for _ in range(config.batch_size * config.gpus):
            self.data.append((start, math.sin(start)))
            start += step

    def next_batch(self, gpu_index):
        start = gpu_index * self.config.batch_size
        return self.data[start: start + self.config.batch_size]


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
            feed_dict = {
                self.tensors.lr: self.config.lr
            }

            for gpu_index in range(self.config.gpus):
                data = self.samples.next_batch(gpu_index)
                feed_dict[self.tensors.s_s[gpu_index]] = data
                feed_dict[self.tensors.z_s[gpu_index]] = np.random.uniform(size=[self.config.batch_size, 1])

            _, loss1 = self.session.run([self.tensors.train_op1, self.tensors.loss1], feed_dict)
            _, loss2 = self.session.run([self.tensors.train_op2, self.tensors.loss2], feed_dict)
            _, loss3 = self.session.run([self.tensors.train_op3, self.tensors.loss3], feed_dict)
            print('%d, loss1: %.8f; loss2: %.8f; loss3: %.8f' % (epoch, loss1, loss2, loss3))

            self.saver.save(self.session, self.config.save_path)

    def predict(self):
        feed_dict = {
            self.tensors.z_s[0]: np.random.uniform(size=[self.config.batch_size, 1])
        }
        fake = self.session.run(self.tensors.fake_s[0], feed_dict)
        for f in fake:
            print('%.8f, %.8f, %.8f' % (f[0], f[1], math.sin(f[0])))


def get_gpus():
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    print(devices, flush=True)
    return len(devices.split(','))


if __name__ == '__main__':
    # Tensors(Config())
    gansin = GANSin()

    gansin.train()
    gansin.predict()
