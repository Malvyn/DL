import tensorflow as tf
v1=tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')
v2=tf.Variable(tf.constant(2),name='v2')
v3=tf.Variable(tf.ones([4,3]),name='v3')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(v1))
    print(sess.run(v2))
    print(sess.run(v3))