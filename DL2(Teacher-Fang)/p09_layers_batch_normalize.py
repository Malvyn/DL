# coding=utf-8

import tensorflow as tf


x = tf.placeholder(tf.float32, [100, 4, 4, 3])

y = tf.layers.batch_normalization(x, axis=[1, 2, 3])

for var in tf.global_variables():
    if not var.trainable:
        print(var.name)
        print([None if sh.value is None else sh.value for sh in var.shape])
        print('-' * 200)
