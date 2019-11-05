# coding=utf-8

import tensorflow as tf
import numpy as np


a = tf.random_uniform([10], -5.0, 5.0, tf.float32)
b = tf.nn.l2_normalize(a)

with tf.Session() as session:
    av, bv = session.run([a, b])

print('l2_normalize on a vector of length 10')
print(av)
print('-' * 200)
print(bv)
print('-' * 200)


a = tf.random_uniform([10, 2], -5.0, 5.0, tf.float32)
b1 = tf.nn.l2_normalize(a)
b2 = tf.nn.l2_normalize(a, axis=-1)

with tf.Session() as session:
    av, b1v, b2v = session.run([a, b1, b2])

print('l2_normalize on a matrix of shape [10, 2]')
print(av)
print('-' * 200)
print(b1v)
print('-' * 200)
print(b2v)
print('-' * 200)


a = tf.random_uniform([10, 20], -5.0, 5.0, tf.float32)
mean = tf.reduce_mean(a, axis=1, keepdims=True)
std = tf.sqrt(tf.reduce_mean(tf.square(a - mean)))


b1 = tf.nn.l2_normalize(a, axis=-1)
b2 = (a - mean) / tf.maximum(1e-10, std)
with tf.Session() as session:
    av, b1v, b2v = session.run([a, b1, b2])

print('Standard normalize on a matrix of shape [10, 2]')
print(av)
print('-' * 200)
print(b1v)
print('-' * 200)
print(b2v)
print('-' * 200)

mean = np.mean(b2v, axis=1)
print(mean)
