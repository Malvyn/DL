import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from skimage.io import imsave

image_height = 28
image_weight = 28
image_size = image_height * image_weight
h1_size = 150  # 第一个隐层神经元数目为150
h2_size = 300  # 第二个隐层神经元数目为300
z_size = 100  # 每个输入层神经元数目为100
batch_size = 256  # 批次
max_epoch = 500  # 最大训练次数



# 讲生成的数据转换为图片
def show_result(x_gen_val, fname):
    x_gen_val = 0.5 * x_gen_val.reshape((x_gen_val.shape[0], image_height, image_weight)) + 0.5
    _, img_h, img_w = x_gen_val.shape
    grid_h = img_h * 8 + 5 * (8 - 1)
    grid_w = img_w * 8 + 5 * (8 - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(x_gen_val):
        if i >= 8 * 8:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // 8) * (img_h + 5)
        col = (i % 8) * (img_w + 5)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)


# 生成网络G
# z_prior: 输入的噪音数据, 是一个Tensor对象, n行z_size列的数据，n行表示n个样本输入，z_size表示每个输入的样本的维度大小
def build_generator(z_prior):
    # 第一层
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name='g_w1', dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h1_size]), name='g_b1', dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)
    # 第二层
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name='g_w2', dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_size]), name='g_b2', dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

    w3 = tf.Variable(tf.truncated_normal([h2_size, image_size], stddev=0.1), name='g_w3', dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([image_size]), name='g_b3', dtype=tf.float32)
    h3 = tf.nn.tanh(tf.matmul(h2, w3) + b3)
    return h3, [w1, b1, w2, b2, w3, b3]

# 判别网络
# x_data 真实数据, x_generated生成数据
def build_disciminator(x_data, x_generated, keep_prob):
    # 1. 合并训练数据
    x_input = tf.concat([x_data, x_generated], axis=0)
    # 第一层
    w1 = tf.Variable(tf.truncated_normal([image_size, h2_size], stddev=0.1), name='d_w1', dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]), name='d_b1', dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_input, w1) + b1), keep_prob)
    # 第二层
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name='d_w2', dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]), name='d_b2', dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)

    # 4. 第三层：Hidden Layer(准备输出)
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name='d_w3', dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name='d_b3', dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3

    # 取出真图片计算后的结果及生成图片计算后的结果 结果用概率的方式输出
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1]))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1]))

    return y_data, y_generated, [w1, b1, w2, b2, w3, b3]

def train():
    # 1. 加载数据(真实数据)
    mnist = input_data.read_data_sets('datas/mnist', one_hot=True)
    # 2. 网络的占位符(训练数据)变量定义
    x_data = tf.placeholder(tf.float32, [batch_size, image_size], name='x_data')
    # x_data_value = 2 * x_data.astype(np.float32) - 1
    # 噪音数据
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name='z_prior')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # 3. 构建生成模型
    x_generated, g_params = build_generator(z_prior)
    # 4. 构建判别模型
    y_data, y_generated, d_params = build_disciminator(x_data, x_generated, keep_prob)
    # 5. 损失函数
    # d_loss = -(tf.log(y_data) + tf.log(1 - y_generated))
    # g_loss = -tf.log(y_generated)

    d_loss = -(tf.reduce_mean(tf.log(y_data)) + tf.reduce_mean(tf.log(1 - y_generated))) / 2
    g_loss = tf.reduce_mean(tf.log(1 - y_generated))

    # 6. 定义优化函数
    optimizer = tf.train.AdamOptimizer(0.0001)
    # var_list: 表示给定更新那些参数
    d_trainer = optimizer.minimize(d_loss, var_list=d_params)
    g_trainer = optimizer.minimize(g_loss, var_list=g_params)

    sess = tf.Session()
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    steps = int(60000 / batch_size)
    for i in range(sess.run(global_step), max_epoch):
        for j in range(steps):
            print("Epoch:{}, Steps:{}".format(i, j))
            x_data_value, _ = mnist.train.next_batch(batch_size)
            x_data_value = 2 * x_data_value.astype(np.float32) - 1
            z_value = np.random.normal(0, 1, size=[batch_size, z_size]).astype(np.float32)
            sess.run(d_trainer, feed_dict={x_data: x_data_value, z_prior: z_value, keep_prob: 0.7})
            sess.run(g_trainer, feed_dict={x_data: x_data_value, z_prior: z_value, keep_prob: 0.7})
            _, g_loss_ = sess.run([d_trainer, d_loss],
                                  feed_dict={x_data: x_data_value, z_prior: z_value, keep_prob: 0.7})
            print("Steps:{}, g_loss:{}".format(j, g_loss_))

            x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
            show_result(x_gen_val, "output_sample/random_sample{}.jpg".format(i))
    sess.close()

train()































