import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    with tf.Graph().as_default():
        num_units = 128  # 隐层的神经元个数为128
        time_steps = 28  # 28时刻
        input_sizes = 28  #28维度（特征）
        n_class = 10 #类别为10个
        learning_rate = 0.001 #学习率
        max_epoch = 1000 #最大迭代次数
        # 一、构建执行图
        global_step_tensor = tf.train.get_or_create_global_step()
        with tf.variable_scope('network', initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)):
            with tf.variable_scope("InputLayer"):
                _X = tf.placeholder(tf.float32, [None, time_steps * input_sizes])
                _Y = tf.placeholder(tf.float32, [None, n_class])
                batch_size = tf.placeholder(tf.int32, [])
                #  将输入的数据转换为给定的要求格式
                inputs = tf.split(_X, num_or_size_splits=time_steps, axis=1)
            with tf.variable_scope("Hidden_Layer"):
                cell = tf.nn.rnn_cell.BasicRNNCell(num_units=num_units)
                initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                output, state = tf.nn.static_rnn(cell, inputs=inputs, initial_state=initial_state)
                # output = output[-1]
                # 所有输出的平均值
                tmp = tf.zeros_like(output[0], dtype=tf.float32)
                for o in output:
                    tmp = tmp + o
                output = tmp / time_steps

            with tf.variable_scope("Output_Layer"):
                w = tf.get_variable('w', shape=[num_units, n_class])
                b = tf.get_variable('b', shape=[n_class])
                logits = tf.add(tf.matmul(output, w), b)
                prediction = tf.argmax(logits, axis=1)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_Y, logits=logits),
                                  name='loss')
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss=loss)
        with tf.name_scope('accuracy'):
            label_ = tf.argmax(_Y, axis=1)
            tmp = tf.cast(tf.equal(prediction, label_), tf.float32)
            accuracy = tf.reduce_mean(tmp)
        # 二、执行图的训练运行
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mnist = input_data.read_data_sets(train_dir='datas/mnist', one_hot=True)
            _batch_size = 64
            for batch in range(max_epoch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=_batch_size)
                _, loss_, accuracy_, = sess.run([train_op, loss, accuracy],
                                                feed_dict={
                                                    _X: batch_x,
                                                    _Y: batch_y,
                                                    batch_size: _batch_size
                                                })
                print("第{}次训练后模型的损失函数为:{}, 准确率:{}".format(batch, loss_, accuracy_))























