# coding=utf-8


import tensorflow as tf


def test_tf():
    a = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    b = a * 0.5 + 3

    c = tf.random_uniform([2, 3, 4])
    d = tf.random_uniform([2, 1, 4])

    with tf.Session() as session:
        c_value, d_value, value = session.run([c, d, c+d])
        print(c_value)
        print('-' * 200)
        print(d_value)
        print('-' * 200)
        print(value)


def build_graph(lr=0.01):
    a = tf.placeholder(tf.float32, [], 'a')
    x = tf.get_variable('x', dtype=tf.float32, initializer=1.0)
    y = x ** 2 - a
    loss = y ** 2

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)

    return a, x, train_op


def train_graph(a, x, train_op, a_value, epoches=1000):
    session = tf.Session()
    with session:
        session.run(tf.global_variables_initializer())
        for _ in range(epoches):
            _, x_value = session.run([train_op, x], feed_dict={a: a_value})
    return x_value


if __name__ == '__main__':
    a, x, train_op = build_graph()
    for a_value in range(2, 11):
        print('sqrt(%s) = %s' % (a_value, train_graph(a, x, train_op, a_value)))
