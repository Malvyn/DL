import numpy as np
import tensorflow as tf
# #输入
# inputs = tf.placeholder(dtype=tf.float32, shape=[4,64])
# #
# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10, activation=tf.nn.tanh)
# state0 = cell.zero_state(batch_size=4, dtype=tf.float32)
#
# output1, state1 = cell.call(inputs=inputs, state=state0)
# output2, state2 = cell.call(inputs=inputs, state=state1)
#
# print("输出为形状:{}".format(output2.get_shape()))
# print("细胞状态为形状:{}".format(state2.get_shape()))


cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10, activation=tf.nn.tanh)


# batch_size = tf.placeholder_with_default(4, shape=[])

inputs1 = tf.placeholder(dtype=tf.float32, shape=[4, 2])
inputs2 = tf.placeholder(dtype=tf.float32, shape=[4, 2])
inputs = [inputs1, inputs2]
state0 = cell.zero_state(batch_size=4, dtype=tf.float32)
output1, state1 = cell.call(inputs=inputs[0], state=state0)
output2, state2 = cell.call(inputs=inputs[1], state=state1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data1 = [
        [1.0, 2.0],
        [2.0, 3.0],
        [4.5, 6.0],
        [9.0, 5.2]
    ]

    data2 = [
        [4.5, 6.0],
        [9.0, 5.2],
        [1.5, 2.3],
        [5.2, 1.3]
    ]

    print(sess.run([output1, output2, state0, state1, state2], feed_dict={
        inputs1: data1,
        inputs2: data2
    }))





























