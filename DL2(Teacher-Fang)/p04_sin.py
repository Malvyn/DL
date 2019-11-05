# coding=utf-8

import tensorflow as tf
import math
import matplotlib.pyplot as pyplot


def get_samples(batch_size):
    start = -math.pi
    end = math.pi

    seg = (end - start) / batch_size

    xs = []
    ys = []
    for _ in range(batch_size):
       xs.append(start)
       ys.append(math.sin(start))
       start += seg

    return xs, ys


def train(batch_size):
    xs, ys = get_samples(batch_size)
    ys_predict = predict(xs, ys)

    pyplot.plot(xs, ys)
    pyplot.plot(xs, ys_predict)
    pyplot.show()


def predict(xs, ys, lr=0.001, epoches=2000, hidden_layers=1, neurals=200):
    x = tf.placeholder(tf.float32, [None], 'x')
    tensor_x = x
    x = tf.reshape(x, [-1, 1])
    for i in range(hidden_layers):
        x = tf.layers.dense(x, neurals, activation=tf.nn.relu, name='dense_%d' % i)

    y_predict = tf.layers.dense(x, 1, name='dense_3')
    y_predict = tf.reshape(y_predict, [-1])
    y = tf.placeholder(tf.float32, [None], 'y')

    loss = tf.reduce_mean(tf.square(y - y_predict))
    opt = tf.train.AdamOptimizer(lr)
    train_op = opt.minimize(loss)
    loss = tf.sqrt(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(epoches):
            feed_dict = {
                tensor_x: xs,
                y: ys
            }
            _, loss_value, y_predict_value = session.run([train_op, loss, y_predict], feed_dict)
            print(loss_value, flush=True)

    return y_predict_value



if __name__ == '__main__':
    train(200)
