import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('datas/mnist', one_hot=True)

train_img = mnist.train.images
train_label = mnist.train.labels


# 定义网络的超参数
learn_rate_base = 1.0
batch_size = 64
display_step = 1
#输入的样本维度大小信息
input_dim = train_img.shape[1]
#输出的维度大小信息
n_classes = train_label.shape[1]

x = tf.placeholder(tf.float32, shape=[None, input_dim], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='y')
learn_rate = tf.placeholder(tf.float32, name='learn_rate')


def get_variable(name, shape=None, dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0, stddev=0.1)):
    return tf.get_variable(name, shape, dtype, initializer)

def learn_rate_func(epoch):  # 更新学习率
    return learn_rate_base * (0.9 ** int(epoch / 10))

def le_net(x, y):
    # 1. 输入层
    with tf.variable_scope('input1'):
        net = tf.reshape(x, shape=[-1, 28, 28, 1])
    # 2. 卷积层
    with tf.variable_scope('conv2'):
        net = tf.nn.conv2d(input=net, filter=get_variable('w', [5, 5, 1, 20]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [20]))
        net = tf.nn.relu(net)
    # 3. 池化
    with tf.variable_scope('pool3'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('conv4'):
        net = tf.nn.conv2d(input=net, filter=get_variable('w', [5, 5, 20, 50]), strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, get_variable('b', [50]))
        net = tf.nn.relu(net)
    with tf.variable_scope('pool5'):
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('fc6'):
        net = tf.reshape(net, shape=[-1, 7 * 7 * 50])
        net = tf.add(tf.matmul(net, get_variable('w', [7 * 7 * 50, 500])), get_variable('b', [500]))
        net = tf.nn.relu(net)
    with tf.variable_scope('fc7'):
        net = tf.add(tf.matmul(net, get_variable('w', [500, n_classes])), get_variable('b', [n_classes]))
        act = tf.nn.softmax(net)
    return act


















