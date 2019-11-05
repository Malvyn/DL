import numpy as np
import tensorflow as tf

x = tf.reshape(tf.range(1, 13, 1), shape=[-1, 6])

x2, x3, x4 = tf.split(x, num_or_size_splits=2, axis=1)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(x2))
    print(sess.run(x3))
