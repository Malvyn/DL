import numpy as np
import tensorflow as tf

#定义细胞
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10, activation=tf.nn.tanh)

# 定义一个输入, 总时刻为2个时刻，每个时刻输入4个样本，每个样本由2维的数据组成

old_inputs = tf.placeholder(tf.float32, shape=[8, 2])

inputs = tf.split(old_inputs, num_or_size_splits=2, axis=0)

s0 = cell.zero_state(batch_size=4, dtype=tf.float32)

output, state = tf.nn.static_rnn(cell=cell, inputs=inputs, initial_state=s0)

print(output)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    data = [
        [1.0, 2.0],
        [2.0, 3.0],
        [4.5, 6.0],
        [9.0, 5.2],
        [4.5, 6.0],
        [9.0, 5.2],
        [1.5, 2.3],
        [5.2, 1.3]
    ]
    # print(sess.run([output, state], feed_dict={
    #     old_inputs: data
    # }))














































