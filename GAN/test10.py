import tensorflow as tf
x = tf.reshape(tf.range(1, 11, dtype=tf.float32), shape=(-1, 1))
y = tf.nn.sigmoid(x)
x1 = tf.slice(y, [0, 0], [5, -1])
x2 = tf.slice(y, [5, 0], [-1, -1])
loss1 = tf.log(x1)
loss2 = tf.log(1 - x2)
loss = -(tf.reduce_sum(loss1) + tf.reduce_sum(loss2)) / 10
loss_ = -(tf.reduce_mean(loss1) + tf.reduce_mean(loss2)) / 2
with tf.Session() as sess:
    print(sess.run(loss1))
    print(sess.run(loss2))
    print(sess.run(loss))
    print(sess.run(loss_))
