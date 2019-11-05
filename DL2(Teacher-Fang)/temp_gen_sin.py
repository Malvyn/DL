# coding=utf-8

import tensorflow as tf
import math
import matplotlib.pyplot as pyplot
import numpy as np

def get_samples(batch_size):
    start = -math.pi
    end = math.pi

    seg = (end - start) / batch_size

    ys = []
    for _ in range(batch_size):
       ys.append((start, math.sin(start)))
       start += seg

    return [e for e in range(len(ys))], ys


def train(batch_size=200):
    xs, ys = get_samples(batch_size)
    ps = predict(xs, ys, samples=batch_size, )

    ps = sorted(ps, key=lambda p: p[0])
    xxs, yys = [], []
    loss = 0
    for x, y in ps:
        if -math.pi <= x <= math.pi:
            xxs.append(x)
            yys.append(y)
            loss += abs(y - math.sin(x)) / abs(y)
            print('x: %12.8f, sin(x): %12.8f, predict: %12.8f' % (x, math.sin(x), y))
    loss /= len(xxs)
    print('test loss: %s' % loss)

    # pyplot.plot(xs, ys)
    pyplot.plot(xxs, yys)
    pyplot.show('sin')


def predict(xs, ys, lr=0.0005, epoches=5000, hidden_layers=1, samples=200, neurals=200):
    y = tf.placeholder(tf.float32, [None, 2], 'y')
    x = tf.placeholder(tf.int32, [None], 'z')

    w = tf.get_variable('w', [samples, 1], tf.float32)
    emb = tf.nn.embedding_lookup(w, x)
    tx = emb
    for i in range(hidden_layers):
        tx = tf.layers.dense(tx, neurals, activation='relu', name='dense_%d' % i)

    y_predict = tf.layers.dense(tx, 2, name='dense_3')
    mean = tf.reduce_mean(w)
    std  = tf.sqrt(tf.reduce_mean(w**2) - mean**2)

    loss = tf.reduce_mean(tf.square(y - y_predict))
    opt = tf.train.AdamOptimizer(lr)
    train_op = opt.minimize(loss)
    loss = tf.sqrt(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for _ in range(epoches):
            feed_dict = {
                x: xs,
                y: ys
            }
            _, loss_value = session.run([train_op, loss], feed_dict)
            print(loss_value, flush=True)

        # for var in tf.trainable_variables():
        #     print(var.name)
        #     print(session.run(var))
        #     print('-' * 200)
        #
        v_mean, v_std = session.run([mean, std])
        print('mean: %.8f, std: %.8f' % (v_mean, v_std), flush=True)
        v_emb = np.random.normal(v_mean, v_std, [samples, 1])

        y_predict_value = session.run(y_predict, {emb: v_emb})
    return y_predict_value


if __name__ == '__main__':
    train()
