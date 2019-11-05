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
        self.save_path = 'models/cyclegan_cgan/cyclegan_cgan'
        self.eps = 1e-5
        self.gpus = get_gpus()
        self.logdir = '../log/cyclegan/'
        self.image_size = 224
        self.layers = 4

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
        self.training = tf.placeholder(tf.bool, [], 'training')
        opt = tf.train.AdamOptimizer(self.lr)
        with tf.variable_scope('cyclegan'):
            for gpu_index in range(config.gpus):
                with tf.device('/gpu:%d' % gpu_index):
                    sub_tensors = SubTensors(self.config, gpu_index, opt, self.training)
                    self.sub_tensors.append(sub_tensors)
                    tf.get_variable_scope().reuse_variables()
        self.train_op = []
        self.loss = []
        with tf.device('/gpu:0'):
            with tf.variable_scope('cyclegan_train'):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = self._get_sub_attrs('gradients')  # [gpus, 2, 4] ==> (gradient, var) * N
                    grads = np.reshape(grads, [-1, 8])

                    for i in range(8):
                        grad = avg_gradients(grads[:, i])
                        self.train_op.append(opt.apply_gradients(grad))

                    losses = self._get_sub_attrs('losses')  # [gpus, 2, 4]
                    losses = np.reshape(losses, [-1, 8])
                    for i in range(8):
                        loss = tf.reduce_mean([e for e in losses[:, i]])
                        self.loss.append(loss)
                        tf.summary.scalar('loss_%d_%d' % (i // 4, i % 4), loss)
            self.summary_op = tf.summary.merge_all()

    def _get_sub_attrs(self, name):
        attrs = []
        for sub_tensor in self.sub_tensors:
            attr = getattr(sub_tensor, name)
            attrs.append(attr)
        return attrs


class SubTensors:
    def __init__(self, config: Config, gpu_index: int, opt, training):
        self.config = config
        self.training = training

        a_sample = tf.placeholder(
            tf.float32, [None, config.image_size, config.image_size, 3], name='a_sample')
        b_sample = tf.placeholder(
            tf.float32, [None, config.image_size, config.image_size, 3], name='b_sample')

        self.samples = (a_sample, b_sample)
        self.fakes = [None, None]
        self.losses = [[None] * 4, [None] * 4]
        self.gradients = [[None] * 4, [None] * 4]  # [2, 4]
        self.gan(0, opt)
        self.gan(1, opt)

    def gan(self, sample_index, opt):
        with tf.variable_scope('generator%d' % sample_index, reuse=tf.AUTO_REUSE):
            fake = generator(self.samples[sample_index], self.config.layers)
            self.fakes[sample_index] = fake * 255

        with tf.variable_scope('discriminator%d' % sample_index):
            sample_p = discriminator(self.samples[sample_index], self.training)
            tf.get_variable_scope().reuse_variables()
            fake_p = discriminator(fake, self.training)

        with tf.variable_scope('generator%d' % (1 - sample_index), reuse=tf.AUTO_REUSE):
            sample_fake = generator(fake, self.config.layers)

        with tf.variable_scope('loss_gradient%d' % sample_index):
            vars = tf.trainable_variables()
            name = 'discriminator%d' % sample_index
            vars_disc = [var for var in vars if name in var.name]
            name = 'generator%d' % sample_index
            vars_genr = [var for var in vars if name in var.name]

            sample_p = tf.maximum(sample_p, self.config.eps)
            self.losses[sample_index][0] = -tf.reduce_mean(tf.log(sample_p))
            self.gradients[sample_index][0] = opt.compute_gradients(self.losses[sample_index][0], vars_disc)

            neg_fake_p = tf.maximum(1 - fake_p, self.config.eps)
            self.losses[sample_index][1] = -tf.reduce_mean(tf.log(neg_fake_p))
            self.gradients[sample_index][1] = opt.compute_gradients(self.losses[sample_index][1], vars_disc)

            fake_p = tf.maximum(fake_p, self.config.eps)
            self.losses[sample_index][2] = -tf.reduce_mean(tf.log(fake_p))
            self.gradients[sample_index][2] = opt.compute_gradients(self.losses[sample_index][2], vars_genr)

            self.losses[sample_index][3] = tf.reduce_mean(tf.square(self.samples[sample_index] - sample_fake))

            self.gradients[sample_index][3] = opt.compute_gradients(self.losses[sample_index][3],
                                                                    [var for var in vars if 'generator' in var.name])


def generator(x, layers):
    return unet(x, layers)


def unet(x, layers):  # [224, 224, 3]
    x = tf.layers.conv2d(x, 64, 7, 2, 'same', name='conv1')  # [112, 112, 64]
    x = tf.nn.relu(x)
    filters = 64

    save = []
    for i in range(layers):
        filters *= 2
        x = tf.layers.conv2d(x, filters, 3, 2, 'same', name='conv2_%d' % i)
        x = tf.nn.relu(x)
        save.append(x)
    # [7, 7, 1024]
    x = tf.layers.average_pooling2d(x, 7, 7)  # [-1, 1, 1, 1024]

    y = tf.layers.conv2d_transpose(x, 1024, 7, 1, 'valid', name='deconv1')
    # [7, 7, 1024]
    y = tf.nn.relu(y)
    save = reversed(save)
    filters = 1024
    for i, s in enumerate(save):
        y += s
        filters //= 2
        y = tf.layers.conv2d_transpose(y, filters, 3, 2, 'same', name='deconv2_%d' % i)
        y = tf.nn.relu(y)
    # [112, 112, 64]
    y = tf.layers.conv2d_transpose(y, 3, 7, 2, 'same', name='deconv3')  # [224, 224, 3]
    return y


def discriminator(x, training):
    return resnet(x, training)


def resnet(x, training):  # [224, 224, 3]
    x = tf.layers.conv2d(x, 64, 7, 2, 'same', name='conv1')  # [112, 112, 64]
    x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=training, name='bn1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, 2, 2)  # [56, 56, 64]

    filters = 64
    for i, num in enumerate([3, 4, 6, 3]):
        for j in range(num):
            resize = j == 0 and i > 0
            with tf.variable_scope('resnet_module%d_%d' % (i, j)):
                x = resnet_module(x, resize, filters, training)
        filters *= 2

    x = tf.layers.average_pooling2d(x, 7, 7)  # [-1, 1, 1, 64*32=2048]
    x = tf.layers.flatten(x)  # [-1, 2048]
    x = tf.layers.dense(x, 1, name='dense')
    x = tf.nn.sigmoid(x)
    return x


def resnet_module(x, resize, filters, training):
    left = tf.layers.conv2d(x, filters, 1, 2 if resize else 1, 'same', name='conv1')
    left = tf.layers.batch_normalization(left, axis=[1, 2, 3], training=training, name='bn1')
    left = tf.nn.relu(left)
    left = tf.layers.conv2d(left, filters, 3, 1, 'same', name='conv2')
    left = tf.layers.batch_normalization(left, axis=[1, 2, 3], training=training, name='bn2')
    left = tf.nn.relu(left)
    filters *= 4
    left = tf.layers.conv2d(left, filters, 1, 1, 'same', name='conv3')
    left = tf.layers.batch_normalization(left, axis=[1, 2, 3], training=training, name='bn3')

    right = x
    if resize or x.shape[-1].value != filters:
        right = tf.layers.conv2d(right, filters, 3, 2 if resize else 1, 'same', name='conv4')
        right = tf.layers.batch_normalization(right, axis=[1, 2, 3], training=training, name='bn4')

    x = left + right
    x = tf.nn.relu(x)
    return x


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
        self.num = 1000
        self.sample_a = np.random.random([self.num, config.batch_size, config.image_size, config.image_size, 3])
        self.sample_b = np.random.random([self.num, config.batch_size, config.image_size, config.image_size, 3])
        self.pos = 0

    def next_batch(self):
        sample_a = self.sample_a[self.pos]
        sample_b = self.sample_b[self.pos]
        self.pos += 1
        if self.pos >= self.num:
            self.pos = 0
        return sample_a, sample_b


class CycleGAN:
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
                    self.tensors.lr: self.config.lr,
                    self.tensors.training: True
                }
                for gpu_index in range(self.config.gpus):
                    sample_a, sample_b = self.samples.next_batch()
                    feed_dict[self.tensors.sub_tensors[gpu_index].samples[0]] = sample_a
                    feed_dict[self.tensors.sub_tensors[gpu_index].samples[1]] = sample_b

                losses = []
                for i in range(8):
                    _, loss = self.session.run([self.tensors.train_op[i], self.tensors.loss[i]], feed_dict)
                    losses.append(loss)

                summary = self.session.run(self.tensors.summary_op, feed_dict)
                file_writer.add_summary(summary, epoch * steps_per_epoch + step)
                print('%d, %s' % (epoch, losses))

            self.saver.save(self.session, self.config.save_path)

    def predict(self):
        pass


def get_gpus():
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    print(devices, flush=True)
    return len(devices.split(','))


if __name__ == '__main__':
    # Tensors(Config())
    app = CycleGAN()

    app.train()
    # app.predict()
