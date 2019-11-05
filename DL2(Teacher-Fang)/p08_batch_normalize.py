#  code=utf-8

import tensorflow as tf
import numpy as np

tf.layers.batch_normalization

def my_batch_normalize(x, name, momentum=0.99, training=True, eps=1e-10, gamma=True, beta=True):   #  x.shape = [-1, h, w, c]
    h, w, c = x.shape[1].value, x.shape[2].value, x.shape[3].value
    with tf.variable_scope(name):
        mom_mean = tf.get_variable('mom_mean', [h, w, c], tf.float32, tf.initializers.zeros, trainable=False)
        mom_std = tf.get_variable('mom_std', [h, w, c], tf.float32, tf.initializers.ones, trainable=False)

        def true_training():
            mean = tf.reduce_mean(x, axis=0)  # [h, w, c]
            std = tf.reduce_mean(x ** 2, axis=0)  # [h, w, c]
            std = tf.sqrt(std - mean ** 2)  # [h, w, c]

            update_mean = tf.assign(mom_mean, momentum * mom_mean + (1 - momentum) * mean)
            update_std = tf.assign(mom_std, momentum * mom_std + (1 - momentum) * std)

            with tf.control_dependencies([update_std, update_mean]):
                u_std = tf.maximum(mom_std, eps)
                return (x - mom_mean) / u_std

        def false_training():
            m_std = tf.maximum(mom_std, eps)
            return (x - mom_mean) / m_std

        x = tf.cond(training, true_training, false_training)

        if gamma:
            g = tf.get_variable('gamma', [h, w, c], tf.float32, tf.initializers.ones)
            x = g * x

        if beta:
            b = tf.get_variable('beta', [h, w, c], tf.float32, tf.initializers.zeros)
            x += b

        return x


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 4, 4, 2], 'x')
    training = tf.placeholder(tf.bool, [], 'training')
    y = my_batch_normalize(x, 'my_bn', training=training)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for i in range(10):
            print('=============== %d ==================' % i)
            xs = np.random.random([2, 4, 4, 2])
            ys = session.run(y, {x: xs, training: True})
            # print(xs)
            # print('-' * 200)
            # print(ys)
            # print('-' * 200)

            for var in tf.global_variables():
                if not var.trainable:
                    print(var.name)
                    print(session.run(var))
                    print('-' * 200)
