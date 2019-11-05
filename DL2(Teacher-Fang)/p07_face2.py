#  coding=utf-8


import tensorflow as tf
import numpy as np
import argparse
# import cv2
#
# img = cv2.imread()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('', img)


class Config:
    def __init__(self):
        self.lr = 0.0001
        self.epoches = 200
        self.batch_size = 20
        self.resnet='3,4,6,3'  # Resnet50
        self.persons = 20  # should be 90000 in practice
        self.save_path = 'models/p07_face/p07_face'  # The first p97_face is sub_directory, the 2nd one is model name
                                                     # instead of file name or sub_directory name.
        self.eps = 1e-10

    def _get_attrs(self):
        result = {}
        for name in dir(self):
            if not name.startswith('_'):
                value = getattr(self, name)
                if type(value) in (int, float, str, bool):
                    result[name] = value
        return result

    def __repr__(self):
        attrs = self._get_attrs()
        result = ''
        for name in attrs:
            value = attrs[name]
            result += '%s: %s, ' % (name, value)
        return result

    def from_cmd_line(self):
        parser = argparse.ArgumentParser()
        attrs = self._get_attrs()
        for name in attrs:
            value = attrs[name]
            parser.add_argument('--' + name, type=type(value), help='Default to %s' % value, default=value)
        a = parser.parse_args()
        for name in attrs:
            setattr(self, name, getattr(a, name))


class Tensors:
    def __init__(self, config):
        with tf.device('/gpu:0'):
            self.training = tf.placeholder(tf.bool, [], name='training')
            self.x = tf.placeholder(tf.float32, [None, 224, 224, 3], 'x')

            resnet = [int(e) for e in config.resnet.split(',')]
            print(resnet, flush=True)
            y_predict = self.resnet(self.x/255, layers=resnet, persons=config.persons)
            self.y_predict = tf.argmax(y_predict, axis=-1, output_type=tf.int32)

            self.y = tf.placeholder(tf.int32, [None], 'y')
            y = tf.one_hot(self.y, config.persons)

            y_predict = tf.maximum(y_predict, config.eps)
            self.loss = tf.reduce_mean(-tf.reduce_sum(tf.log(y_predict) * y, axis=-1))
            self.lr = tf.placeholder(tf.float32, [], 'lr')
            opt = tf.train.AdamOptimizer(self.lr)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = opt.minimize(self.loss)

            precise = tf.equal(self.y, self.y_predict)
            precise = tf.cast(precise, tf.float32)
            self.precise = tf.reduce_mean(precise)

    def resnet(self, x, layers, persons):
        x = tf.layers.conv2d(x, 64, 7, 2, 'same', name='conv1')  # [112, 112, 64]
        x = tf.layers.batch_normalization(x, axis=[1, 2, 3], training=self.training)
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x, 2, 2)  # [56, 56, 64]

        filters = 64
        for i, num in enumerate(layers):  # [3, 4, 6, 3]
            for j in range(num):
                resize = j == 0 and i > 0
                x = self.resnet_module(x, resize, filters)
            filters *= 2

        x = tf.layers.average_pooling2d(x, 7, 7)  # [1, 1]
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, persons)
        x = tf.nn.softmax(x)

        return x

    def resnet_module(self, x, resize, filters):
        left = tf.layers.conv2d(x, filters, 1, 2 if resize else 1, 'same')
        left = tf.layers.batch_normalization(left, axis=[1, 2, 3], training=self.training)
        left = tf.nn.relu(left)
        left = tf.layers.conv2d(left, filters, 3, 1, 'same')
        left = tf.layers.batch_normalization(left, axis=[1, 2, 3], training=self.training)
        left = tf.nn.relu(left)
        filters *= 4
        left = tf.layers.conv2d(left, filters, 1, 1, 'same')
        left = tf.layers.batch_normalization(left, axis=[1, 2, 3], training=self.training)

        right = x
        if resize or x.shape[-1].value != filters:
            right = tf.layers.conv2d(right, filters, 3, 2 if resize else 1, 'same')
            right = tf.layers.batch_normalization(right, axis=[1, 2, 3], training=self.training)

        x = left + right
        x = tf.nn.relu(x)
        return x


class Samples:
    def __init__(self, config):
        self.num = 1000000
        self.config = config

    def next_batch(self, batch_size):
        xs = np.random.uniform(0, 256, [batch_size, 224, 224, 3])
        ys = np.random.randint(0, self.config.persons, [batch_size])
        return xs, ys


class Face:
    def __init__(self):
        self.config = Config()
        self.samples = Samples(self.config)

        graph = tf.Graph()
        with graph.as_default():
            self.ts = Tensors(self.config)
            self.saver = tf.train.Saver()
            session_config = tf.ConfigProto()
            session_config.allow_soft_placement = True
            self.session = tf.Session(graph=graph, config=session_config)

            try:
                self.saver.restore(self.session, self.config.save_path)
                print('Success to restore the model from %s' % self.config.save_path)
            except:
                print('Fail to restore the model from %s, use a new model instead.' % self.config.save_path)
                self.session.run(tf.global_variables_initializer())

    def train(self):
        config = self.config
        steps_per_epoch = int(self.samples.num / config.batch_size)
        for epoch in range(config.epoches):
            for step in range(steps_per_epoch):
                xs, ys = self.samples.next_batch(config.batch_size)
                feed_dict = {
                    self.ts.x: xs,
                    self.ts.y: ys,
                    self.ts.lr: config.lr,
                    self.ts.training: True
                }
                _, loss = self.session.run([self.ts.train_op, self.ts.loss], feed_dict)

                print('%d/%d: %s' % (step, epoch, loss), flush=True)
            self.saver.save(self.session, config.save_path)

    def predict(self):
        xs, ys = self.samples.next_batch(self.config.batch_size)
        feed_dict = {
            self.ts.x: xs,
            self.ts.y: ys,
            self.ts.training: False
        }
        predict, precise = self.session.run([self.ts.y_predict, self.ts.precise], feed_dict)
        print(predict, flush=True)
        print(ys)
        print('precise = %s' % precise, flush=True)


if __name__ == '__main__':
    face = Face()
    face.config.from_cmd_line()
    face.train()


    tf.layers.conv2d()
