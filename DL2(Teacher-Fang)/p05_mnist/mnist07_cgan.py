# coding=utf-8


import tensorflow as tf
import numpy as np
from types import MethodType, FunctionType
import argparse
import math
import os
import cv2

from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

"""
Use samples of [x, sin(x)] to train a GAN
"""


class Config:
    def __init__(self):
        self.batch_size = 10
        self.lr = 0.0002
        self.epoches = 1000
        self.save_path = 'models/mnist07_cgan/mnist07_cgan'
        self.eps = 1e-5
        self.gpus = get_gpus()
        self.logdir = '../log/mnist07/'
        self.z_size = 16
        self.samples_path = '../../deeplearning_ai12/p07_mnist/MNIST_data'

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
        self.sub_tensors = []
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        opt = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope('mnist07'):
            for gpu_index in range(config.gpus):
                with tf.device('/gpu:%d' % gpu_index):
                    sub_tensors = SubTensors(self.config, gpu_index, opt)
                    self.sub_tensors.append(sub_tensors)
                    tf.get_variable_scope().reuse_variables()
        self.train_op = []
        self.loss = []
        with tf.device('/gpu:0'):
            with tf.variable_scope('mnist07_train'):
                for i in range(1, 4):
                    grad = avg_gradients(self._get_sub_attrs('gradient%d' % i))
                    self.train_op.append(opt.apply_gradients(grad))
                    loss = tf.reduce_mean(self._get_sub_attrs('loss%d' % i))
                    self.loss.append(loss)
                    tf.summary.scalar('loss_%d' % i, loss)

            self.summary_op = tf.summary.merge_all()

    def _get_sub_attrs(self, name):
        attrs = []
        for sub_tensor in self.sub_tensors:
            attr = getattr(sub_tensor, name)
            attrs.append(attr)
        return attrs


class SubTensors:
    def __init__(self, config: Config, gpu_index: int, opt):
        self.sample = tf.placeholder(tf.float32, [None, 28*28], name='sample')
        sample = tf.reshape(self.sample, [-1, 28, 28, 1])

        self.z = tf.placeholder(tf.float32, [None, config.z_size], name='z')
        self.label = tf.placeholder(tf.int32, [None], name='label')
        label = tf.one_hot(self.label, 10)

        with tf.variable_scope('generator'):
            fake = generator(self.z, label)
            self.fake = fake * 255

        with tf.variable_scope('discriminator'):
            sample_p = discriminator(sample, label)
            tf.get_variable_scope().reuse_variables()
            fake_p = discriminator(fake, label)

        with tf.variable_scope('loss_gradient'):
            sample_p = tf.maximum(config.eps, sample_p)
            self.loss1 = -tf.reduce_mean(tf.log(sample_p))

            fake_p = tf.maximum(config.eps, fake_p)
            self.loss2 = -tf.reduce_mean(tf.log(1 - fake_p))
            self.loss3 = -tf.reduce_mean(tf.log(fake_p))

            vars = tf.trainable_variables()
            vars_disc = [var for var in vars if 'discriminator' in var.name]
            vars_genr = [var for var in vars if 'generator' in var.name]

            self.gradient1 = opt.compute_gradients(self.loss1, vars_disc)
            self.gradient2 = opt.compute_gradients(self.loss2, vars_disc)
            self.gradient3 = opt.compute_gradients(self.loss3, vars_genr)


def discriminator(x, label):  #  x.shape: [-1, 28, 28, 1]; label.shape: [-1, 10]
    x = tf.layers.conv2d(x, 32, 3, 1, 'same', name='conv1')  # [-1, 28, 28, 32]
    filters = 32
    size = 28
    for i in range(2):
        filters *= 2
        size //= 2
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv2_%d' % i)
        x = tf.layers.max_pooling2d(x, 2, 2)

        lab = tf.layers.dense(label, size**2 * filters, name='label_dense1_%d' % i)
        lab = tf.reshape(lab, [-1, size, size, filters])
        x += lab
    # [-1, 14, 14, 64]
    # [-1, 7, 7, 128]

    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, 1000, tf.nn.relu, name='dense1')  # [-1, 1000]
    lab = tf.layers.dense(label, 1000, name='label_dense2')
    x += lab
    x = tf.layers.dense(x, 1, name='dense2')  # [-1, 1]
    x = tf.nn.sigmoid(x)
    return x  # [-1, 1]


def generator(z, label):  # z.shape: [-1, 16]; label.shape: [-1, 10]
    z = tf.layers.dense(z, 7*7*128, name='dense1')
    z = tf.reshape(z, [-1, 7, 7, 128]) # [-1, 7, 7, 128]
    filters = 128
    size = 7
    for i in range(2):
        filters //= 2
        size *= 2
        z = tf.layers.conv2d_transpose(z, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv%d' % i)

        lab = tf.layers.dense(label, size * size * filters, name='dense2_%d' % i)
        lab = tf.reshape(lab, [-1, size, size, filters])
        z += lab

    # [-1, 14, 14, 64]
    # [-1, 28, 28, 32]
    z = tf.layers.conv2d_transpose(z, 1, 3, 1, 'same', name='deconv_last')  # [-1, 28, 28, 1]
    lab = tf.layers.dense(label, size**2, name='dense3')
    lab = tf.reshape(lab, [-1, size, size, 1])
    z += lab
    z = tf.nn.sigmoid(z)
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
        self.ds = read_data_sets(config.samples_path)
        self.num = self.ds.train.num_examples

    def next_batch(self, batch_size):
        return self.ds.train.next_batch(batch_size)


class MnistCGAN:
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
        file_writer = tf.summary.FileWriter(self.config.logdir, self.session.graph)

        for epoch in range(self.config.epoches):
            steps_per_epoch = self.samples.num // (self.config.gpus * self.config.batch_size)
            for step in range(steps_per_epoch):
                feed_dict = {
                    self.tensors.lr: self.config.lr
                }
                for gpu_index in range(self.config.gpus):
                    sample, label = self.samples.next_batch(self.config.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].sample] = sample
                    feed_dict[self.tensors.sub_tensors[gpu_index].label] = label
                    feed_dict[self.tensors.sub_tensors[gpu_index].z] = \
                        np.random.normal(size=[self.config.batch_size, self.config.z_size])

                losses = []
                for i in range(3):
                    _, loss = self.session.run([self.tensors.train_op[i], self.tensors.loss[i]], feed_dict)
                    losses.append(loss)

                summary = self.session.run(self.tensors.summary_op, feed_dict)
                file_writer.add_summary(summary, epoch * steps_per_epoch + step)
                print('%d, loss1: %.8f; loss2: %.8f; loss3: %.8f' % (epoch, losses[0], losses[1], losses[2]))

            self.saver.save(self.session, self.config.save_path)

    def predict(self):
        feed_dict = {
            self.tensors.sub_tensors[0].z: np.random.normal(size=[self.config.batch_size, self.config.z_size]),
            self.tensors.sub_tensors[0].label: [e % 10 for e in range(self.config.batch_size)]
        }
        fake = self.session.run(self.tensors.sub_tensorsf[0].fake, feed_dict)  # [-1, 28, 28, 1]
        # fake = np.reshape(fake, [-1, 28, 28])
        # fake = np.concatenate(fake, axis=1)  # [-1, 28]
        fake = np.reshape(fake, [-1, 28])
        cv2.imshow('Image', fake)
        # cv2.imwrite('.....jpg', fake)


def get_gpus():
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    print(devices, flush=True)
    return len(devices.split(','))


if __name__ == '__main__':
    # Tensors(Config())
    app = MnistCGAN()

    app.train()
    # app.predict()
