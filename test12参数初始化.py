import tensorflow as tf

# w = tf.get_variable(name='w', shape=[100], initializer=tf.random_normal_initializer(0.0, 0.1))
w = tf.get_variable(name='w', shape=[100],initializer=tf.truncated_normal_initializer(0.0, 0.1))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))

