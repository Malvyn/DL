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
        self.save_path = 'models/wgan/wgan'
        self.eps = 1e-5
        self.gpus = get_gpus()
        self.logdir = '../log/wgan/'
        self.samples_path = '../deeplearning_ai12/p07_mnist/MNIST_data'
        self.semantics_size = 32
        self.image_path = 'images/p34_wgan.jpg'
        self.c = 0.01

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
        with tf.variable_scope('wgan'):
            for gpu_index in range(config.gpus):
                with tf.device('/gpu:%d' % gpu_index):
                    sub_tensors = SubTensors(self.config, gpu_index, opt)
                    self.sub_tensors.append(sub_tensors)
                    tf.get_variable_scope().reuse_variables()
        self.train_op = []
        self.loss = []
        with tf.device('/gpu:0'):
            with tf.variable_scope('wgan_train'):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    for i in range(1, 5):
                        loss = self._get_sub_attrs('loss%d' % i)
                        loss = tf.reduce_mean(loss)
                        self.loss.append(loss)
                        tf.summary.scalar('loss_%d_%d' % (i // 4, i % 4), loss)

                    vars_disc = [var for var in tf.trainable_variables() if 'discriminator' in var.name]
                    for i in range(1, 5):
                        g = self._get_sub_attrs('gradient%d' % i)  # [N, 2]
                        grad = avg_gradients(g)
                        op = opt.apply_gradients(grad)
                        if i in (1, 2):
                            with tf.control_dependencies([op]):
                                clip = [tf.assign(var, tf.clip_by_value(var, -config.c, config.c)) for var in vars_disc]
                                self.train_op.append(clip)
                        else:
                            self.train_op.append(op)

            self.summary_op = tf.summary.merge_all()

    def _get_sub_attrs(self, name):
        attrs = []
        for sub_tensor in self.sub_tensors:
            attr = getattr(sub_tensor, name)
            attrs.append(attr)
        return attrs


class SubTensors:
    def __init__(self, config: Config, gpu_index: int, opt):
        self.config = config

        self.sample = tf.placeholder(tf.float32, [None, 28*28], name='sample')
        sample = tf.reshape(self.sample, [-1, 28, 28, 1])
        self.sample_label = tf.placeholder(tf.int32, [None], name='sample_label')
        sample_label = tf.one_hot(self.sample_label, 10)
        self.target_label = tf.placeholder(tf.int32, [None], name='target_label')
        target_label = tf.one_hot(self.target_label, 10)

        with tf.variable_scope('generator'):
            fake = self.generator(sample, target_label)
            self.fake = tf.reshape(fake, [-1, 28, 28]) * 255
            tf.get_variable_scope().reuse_variables()
            sample_fake = self.generator(fake, sample_label)

        with tf.variable_scope('discriminator'):
            sample_em = discriminator(sample, sample_label)
            tf.get_variable_scope().reuse_variables()
            fake_em = discriminator(fake, target_label)

        with tf.variable_scope('loss_gradient'):
            vars = tf.trainable_variables()
            vars_disc = [var for var in vars if 'discriminator' in var.name]
            vars_genr = [var for var in vars if 'generator' in var.name]

            self.loss1 = -tf.reduce_mean(sample_em)
            self.gradient1 = opt.compute_gradients(self.loss1, vars_disc)

            self.loss2 = tf.reduce_mean(fake_em)
            self.gradient2 = opt.compute_gradients(self.loss2, vars_disc)

            self.loss3 = -tf.reduce_mean(fake_em)
            self.gradient3 = opt.compute_gradients(self.loss3, vars_genr)

            self.loss4 = tf.reduce_mean(tf.square(sample - sample_fake))
            self.gradient4 = opt.compute_gradients(self.loss4, vars_genr)

    def generator(self, sample, label):
        """
        Generate a fake through the sample as well as the label.
        :param sample: [-1, 28, 28, 1]
        :param label:  [-1, 10]
        :return: [-1, 28, 28, 1]
        """
        filters = 32  # 64
        sample = tf.layers.conv2d(sample, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
        size = 28
        # [-1, 28, 28, 32]
        for i in range(2):
            filters *= 2
            size //= 2
            sample = tf.layers.conv2d(sample, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)

            lab = tf.layers.dense(label, size*size*filters, name='lab_dense1_%d' % i)
            lab = tf.reshape(lab, [-1, size, size,  filters])
            sample += lab

        # [-1, 7, 7, 128]
        sample = tf.layers.flatten(sample)
        sample = tf.layers.dense(sample, self.config.semantics_size, name='dense1')

        y = tf.layers.dense(sample, size*size*filters, name='dense2')
        y = tf.reshape(y, [-1, size, size, filters])
        for i in range(2):
            filters //=2
            size *= 2
            y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', activation=tf.nn.relu, name='deconv1_%d' % i)

            lab = tf.layers.dense(label, size * size * filters, name='lab_dense2_%d' % i)
            lab = tf.reshape(lab, [-1, size, size, filters])
            y += lab

        # [-1, 28, 28, 32]
        y = tf.layers.conv2d_transpose(y, 1, 3, 1, 'same', name='deconv2')  # [-1, 28, 28, 1]
        # y = tf.nn.sigmoid(y)
        return y


def discriminator(fake, label):
    """
    Discriminate whether or not the fake likes a sample.
    :param fake: [-1, 28, 28, 1]
    :param label: [-1, 10]
    :return: [-1, 1]
    """
    filters = 32  # 64
    fake = tf.layers.conv2d(fake, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')
    size = 28
    # [-1, 28, 28, 32]
    for i in range(2):
        filters *= 2
        size //= 2
        fake = tf.layers.conv2d(fake, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)

        lab = tf.layers.dense(label, size * size * filters, name='lab_dense1_%d' % i)
        lab = tf.reshape(lab, [-1, size, size, filters])
        fake += lab

    # [-1, 7, 7, 128]
    fake = tf.layers.flatten(fake)
    fake = tf.layers.dense(fake, 1000, activation=tf.nn.relu, name='dense1')
    fake = tf.layers.dense(fake, 1, name='dense2')  # [-1, 1]
    return fake  # Earth Mover distance


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


class StarGAN:
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
            steps_per_epoch = self.samples.num // self.config.gpus
            for step in range(steps_per_epoch):
                feed_dict = {
                    self.tensors.lr: self.config.lr
                }
                for gpu_index in range(self.config.gpus):
                    sample, label = self.samples.next_batch(self.config.batch_size)
                    feed_dict[self.tensors.sub_tensors[gpu_index].sample] = sample
                    feed_dict[self.tensors.sub_tensors[gpu_index].sample_label] = label
                    feed_dict[self.tensors.sub_tensors[gpu_index].target_label] = \
                        np.random.randint(0, 10, [self.config.batch_size])

                losses = []
                for i in range(4):
                    _, loss = self.session.run([self.tensors.train_op[i], self.tensors.loss[i]], feed_dict)
                    losses.append(loss)

                summary = self.session.run(self.tensors.summary_op, feed_dict)
                file_writer.add_summary(summary, epoch * steps_per_epoch + step)
                print('%d, %s' % (epoch, losses))

            self.saver.save(self.session, self.config.save_path)

    def predict(self):
        feed_dict = {
            self.tensors.lr: self.config.lr
        }
        gpu_index = 0
        sample, label = self.samples.next_batch(self.config.batch_size)
        feed_dict[self.tensors.sub_tensors[gpu_index].sample] = sample
        feed_dict[self.tensors.sub_tensors[gpu_index].target_label] = [e % 10 for e in range(self.config.batch_size)]
        fake = self.session.run(self.tensors.sub_tensors[gpu_index].fake)  # [-1, 28, 28]
        img = np.reshape(fake, [-1, 28])

        cv2.imwrite(self.config.image_path, img)
        # cv2.imshow('StarGAN', img)
        # cv2.waitKey(3000)


def get_gpus():
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    print(devices, flush=True)
    return len(devices.split(','))


if __name__ == '__main__':
    # Tensors(Config())
    app = StarGAN()

    app.train()
    # app.predict()
