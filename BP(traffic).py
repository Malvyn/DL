import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    with tf.Graph().as_default():
        # 一、执行图的构建
        with tf.variable_scope('network', initializer=tf.random_normal_initializer(0, 0.01)):
            with tf.variable_scope('InputLayer'):
                # a. 定义占位符
                input_x = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
                input_y = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='y')
            with tf.variable_scope('HiddenLayer1'):
                w = tf.get_variable(name='w', shape=[3, 100], dtype=tf.float32)
                b = tf.get_variable(name='b', shape=[100], dtype=tf.float32)
                z = tf.matmul(input_x, w) + b
                o = tf.nn.sigmoid(z)
            with tf.variable_scope('HiddenLayer2'):
                w = tf.get_variable(name='w', shape=[100, 10], dtype=tf.float32)
                b = tf.get_variable(name='b', shape=[10], dtype=tf.float32)
                z = tf.matmul(o, w) + b
                o = tf.nn.sigmoid(z)
            with tf.variable_scope('OutputLayer'):
                w = tf.get_variable(name='w', shape=[10, 2], dtype=tf.float32)
                b = tf.get_variable(name='b', shape=[2], dtype=tf.float32)
                y_ = tf.matmul(o, w) + b
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(input_y - y_))
        with tf.name_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            train_op = optimizer.minimize(loss=loss)
        # 二、执行图的运行
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #读数据
            df = pd.read_csv('datas/traffic_data.csv')
            x = df[['人口数', '机动车数', '公路面积']]
            y = df[['客运量', '货运量']]
            scaler_x = MinMaxScaler(feature_range=(-10, 10))
            scaler_y = MinMaxScaler(feature_range=(-10, 10))
            x = scaler_x.fit_transform(x)
            y = scaler_y.fit_transform(y)
            x = np.asarray(x)
            y = np.asarray(y)
            for step in range(10000):
                _, loss_ = sess.run([train_op, loss], feed_dict={
                    input_x: x,
                    input_y: y
                })
                print("第{}次训练后模型的损失函数为:{}".format(step, loss_))
            predict = sess.run(y_, feed_dict={input_x: x})
            print("模型返回的预测值(区间缩放法之后的):")
            print(predict[:2])
            print("还原最终的模型结果值:")
            print(scaler_y.inverse_transform(predict[:2]))

            t = range(len(x))
            plt.plot(t, y[:,0], 'ro')
            plt.plot(t, predict[:,0], 'g-')
            plt.show()





























