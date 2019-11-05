import tensorflow as tf

x = tf.constant([
    [3, 0, 1, 5, 9, 8, 1, 6],
    [5, 5, 2, 4, 7, 1, 9, 4]
])

values = tf.one_hot(indices=x, depth=10)
with tf.Session() as sess:
    print(sess.run(values))

