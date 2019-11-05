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
        self.save_path = 'models/p36_face_change/face_change'
        self.eps = 1e-5
        self.gpus = get_gpus()
        self.logdir = '../log/face_change/'
        self.semantics_size = 32  # should be 200 at least
        self.c = 0.01
        self.image_size = 32  # should be 128 as least
        self.images = 5
        self.filters = 32  # should be 64
        self.convs = 4  # should be 4 or 5

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

        self.sources = tf.placeholder(tf.float32, [None, config.images, config.image_size, config.image_size, 3])
        sources = self.sources / 255

        self.source = tf.placeholder(tf.float32, [None, config.image_size, config.image_size, 3])
        source = self.source / 255

        self.target = tf.placeholder(tf.float32, [None, config.image_size, config.image_size, 3])
        target = self.target / 255

        self.targets = tf.placeholder(tf.float32, [None, config.images, config.image_size, config.image_size, 3])
        targets = self.targets / 255

        with tf.variable_scope('generator'):
            fake = self.generator(sources, target)
            self.fake = fake * 255

            tf.get_variable_scope().reuse_variables()
            target_fake = self.generator(targets, fake)

        with tf.variable_scope('discriminator'):
            sample_em = self.discriminator(sources, source)
            tf.get_variable_scope().reuse_variables()
            fake_em = self.discriminator(sources, fake)

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

            loss4 = tf.reduce_mean(tf.square(target - target_fake))
            self.gradient4 = opt.compute_gradients(loss4, vars_genr)
            self.loss4 = tf.sqrt(loss4)

    def generator(self, sources, target):
        vectors = 0
        with tf.variable_scope('cycle'):
            for i in range(self.config.images):
                source = sources[:, i, :, :, :]  # [-1, image_size, image_size, 3]
                vector = self.encoder(target, source, self.config.semantics_size)
                vectors += vector
                tf.get_variable_scope().reuse_variables()

        result = self.decoder(vectors, target)  # [-1, image_size, image_size, 3]
        return result

    def discriminator(self, sources, x):
        vectors = 0
        with tf.variable_scope('cycle'):
            for i in range(self.config.images):
                source = sources[:, i, :, :, :]  # [-1, image_size, image_size, 3]
                vector = self.encoder(x, source, 1)
                vectors += vector
                tf.get_variable_scope().reuse_variables()
        # [-1, 1]
        return vectors

    def encoder(self, x, label, vector_size):
        """
        Encode the x with the label.
        :param x: [-1, image_size, image_size, 3]
        :param label: [-1, image_size, image_size, 3]
        :return: [-1, semantics_size]
        """
        filters = self.config.filters
        x = tf.layers.conv2d(x, filters, 3, 1, 'same', activation=tf.nn.relu, name='conv1')

        label = tf.layers.conv2d(x, filters, 3, 1, 'same', name='label_conv1')
        x += label

        size = self.config.image_size
        for i in range(self.config.convs):
            filters *= 2
            size //= 2
            x = tf.layers.conv2d(x, filters, 3, 2, 'same', activation=tf.nn.relu, name='conv2_%d' % i)

            label = tf.layers.conv2d(label, filters, 3, 2, 'same', name='label_conv2_%d' % i)
            x += label

        # [2, 2, 512]
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, vector_size, name='dense')
        return x

    def decoder(self, vector, target):
        """

        :param vector: [-1, semantics_size]
        :param target: [-1, image_size, image_size, 3]
        :return:
        """
        filters = self.config.filters * 2 ** self.config.convs  # 32 * 2**4 = 512
        size = self.config.image_size // (2 ** self.config.convs)  # 32 / 2**4 = 2
        vector = tf.layers.dense(vector, size * size * filters, name='dense1')
        vector = tf.reshape(vector, [-1, size, size, filters])

        for i in range(self.config.convs):
            filters //= 2
            size *= 2
            vector = tf.layers.conv2d_transpose(vector, filters, 3, 2, 'same', \
                                                activation=tf.nn.relu, name='deconv1_%d' % i)
            strides = self.config.image_size//size
            tar = tf.layers.conv2d(target, filters, 3, strides, 'same', name='target_deconv2_%d' % i)
            vector += tar
        # [-1, image_size, image_size, 32]

        vector = tf.layers.conv2d_transpose(vector, 3, 3, 1, 'same', name='deconv2')
        return vector


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
        self.num = 100
        self.images = np.random.random([self.num, config.images+1, config.image_size, config.image_size, 3])

    def next_batch(self, batch_size):
        return self.person_random(batch_size), self.person_random(batch_size)

    def person_random(self, batch_size):
        result = []
        for _ in range(batch_size):
            index = np.random.randint(0, self.num)
            result.append(self.images[index])
        return result  # [batch_size, images, image_size, image_size, 3]


class FaceChange:
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
                    sources, targets = self.samples.next_batch(self.config.batch_size)
                    source, sources = self.fetch_one(sources)
                    target, targets = self.fetch_one(targets)
                    feed_dict[self.tensors.sub_tensors[gpu_index].sources] = sources
                    feed_dict[self.tensors.sub_tensors[gpu_index].source] = source
                    feed_dict[self.tensors.sub_tensors[gpu_index].target] = target
                    feed_dict[self.tensors.sub_tensors[gpu_index].targets] = targets

                losses = []
                for i in range(4):
                    _, loss = self.session.run([self.tensors.train_op[i], self.tensors.loss[i]], feed_dict)
                    losses.append(loss)

                summary = self.session.run(self.tensors.summary_op, feed_dict)
                file_writer.add_summary(summary, epoch * steps_per_epoch + step)
                print('%d, %s' % (epoch, losses))

            self.saver.save(self.session, self.config.save_path)

    def predict(self):
        pass

    def fetch_one(self, images):
        """

        :param images: [-1, images+1, image_size, image_size, 3]
        :return:
        """

        fetch_index = np.random.randint(0, self.config.images, [len(images)])
        one = []
        rest = []
        for i, imgs in enumerate(images):
            ind = fetch_index[i]
            fetch = imgs[ind]
            one.append(fetch)
            imgs = np.concatenate((imgs[:ind, :, :, :], imgs[ind+1:, :, :, :]), axis=0)
            rest.append(imgs)
        return one, rest


def get_gpus():
    devices = os.getenv('CUDA_VISIBLE_DEVICES', '0')
    print(devices, flush=True)
    return len(devices.split(','))


if __name__ == '__main__':
    # Tensors(Config())
    app = FaceChange()
    app.train()
    # app.predict()
