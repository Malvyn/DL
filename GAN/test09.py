import numpy as np
import tensorflow as tf

# x = tf.reshape(tf.range(1, 11), shape=(-1, 1))
x = tf.reshape(tf.range(1, 11, dtype=tf.float32), shape=(-1, 1))
y = tf.nn.sigmoid(x)
x1 = tf.slice(y, [0, 0], [7, -1]) # begin:表示从x[0,0],即第一个元素开始 size:表示从x[0,0]开始，对
                                  # x的第一维（行）取7 第二维（列）取-1
x2 = tf.slice(y, [7, 0], [-1, -1])

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(x1))
    print(sess.run(x2))

z_sample_val = np.random.normal(0, 1, size=(1, 100)).astype(np.float32)

print(z_sample_val)

