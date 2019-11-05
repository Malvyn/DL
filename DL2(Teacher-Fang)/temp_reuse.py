import tensorflow as tf

with tf.variable_scope('abc'):
    for i in range(10):
        a = tf.get_variable('a%d' % i, shape=[200, 200], dtype=tf.float32)
        a = tf.layers.dense(a, 10)
        for var in tf.trainable_variables():
            print(var.name)
        print('-' * 200)


    for i in range(10):
        a = tf.get_variable('b', shape=[200, 200], dtype=tf.float32)
        a = tf.layers.dense(a, 10, name='kkk')
        for var in tf.trainable_variables():
            print(var.name)
        print('-' * 200)
        tf.get_variable_scope().reuse_variables()

tf.nn.depthwise_conv2d



