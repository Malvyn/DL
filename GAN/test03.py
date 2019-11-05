import tensorflow as tf

x = tf.reshape(tf.range(1, 7, 1), shape=[-1, 3])

x1 ,x2, x3 = tf.split(x, num_or_size_splits=3, axis=1)

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(x1))
    print(sess.run(x2))
    print(sess.run(x3))
