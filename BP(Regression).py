import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.variable_scope('network',initializer=tf.random_normal_initializer(0, 0.01)):
            with tf.variable_scope('InputLayer'):
                # a. 定义占位符
                input_x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
                input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
            with tf.variable_scope('HiddenLayer'):
                # b. 定义模型参数
                w = tf.get_variable(name='w', shape=[1, 10], dtype=tf.float32)
                b = tf.get_variable(name='b', shape=[10], dtype=tf.float32)
                # c. 模型预测的构建（获取预测值）
                z = tf.matmul(input_x, w) + b
                o = tf.nn.sigmoid(z)
            with tf.variable_scope('OutputLayer'):
                # b. 定义模型参数
                w = tf.get_variable(name='w', shape=[10, 1], dtype=tf.float32)
                b = tf.get_variable(name='b', shape=[1], dtype=tf.float32)
                # c. 模型预测的构建（获取预测值）
                y_ = tf.matmul(o, w) + b
        with tf.name_scope('loss'):
            # d. 损失函数构建（平方和损失函数）
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(input_y - y_), reduction_indices=[1]))
            # loss = tf.reduce_mean(tf.square(input_y - y_))
        with tf.name_scope('train'):
            # e. 定义优化器
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            # f. 定义一个训练操作对象
            train_op = optimizer.minimize(loss=loss)
        with tf.Session() as sess:
            # a. 变量的初始化操作
            sess.run(tf.global_variables_initializer())
            N = 100
            x = np.linspace(0, 6, N) + np.random.normal(0, 2.0, N)
            y = 14 * x + 7 + np.random.normal(0, 5.0, N)
            x.shape = -1, 1
            y.shape = -1, 1
            for step in range(20000):
                # 1. 触发模型训练操作
                _, loss_ = sess.run([train_op, loss], feed_dict={
                    input_x: x,
                    input_y: y
                })
                print("第{}次训练后模型的损失函数为:{}".format(step, loss_))
            predict = sess.run(y_, feed_dict={input_x: x})
            plt.plot(x, y, 'ro')
            plt.plot(x, predict, 'g-')
            plt.show()










