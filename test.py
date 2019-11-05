import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取图片数据，并对标签y做one-hot
minis=input_data.read_data_sets('./data',one_hot=True)
train_img=minis.train.images
train_label=minis.train.labels
test_img=minis.test.images
test_label=minis.test.labels

# print(train_label[0])
# 执行图的构建
with tf.Graph().as_default() :
    with tf.variable_scope('inputlayer'):
        x=tf.placeholder(dtype=tf.float32,shape=[None,784],name='x')
        y=tf.placeholder(dtype=tf.float32,shape=[None,10],name='y')
        net=tf.reshape(x,shape=[-1,28,28,1])
    # 卷积
    with tf.variable_scope('conv1'):
        w=tf.get_variable(name='w',shape=[5,5,1,20],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        net=tf.nn.conv2d(input=net, filter=w, strides=[1,1,1,1], padding='SAME')
        b=tf.get_variable(name='b',shape=[20],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        net=tf.nn.bias_add(net,b)
        net=tf.nn.relu(net)
    # 池化
    with tf.variable_scope('pooling1'):
        net=tf.nn.max_pool(value=net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    # 卷积
    with tf.variable_scope('conv2'):
        w=tf.get_variable(name='w',shape=[5,5,20,50],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        net=tf.nn.conv2d(input=net,filter=w,strides=[1,1,1,1],padding='SAME')
        b=tf.get_variable(name='b',shape=[50],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        net=tf.nn.bias_add(net,b)
        net=tf.nn.relu(net)
    # 池化
    with tf.variable_scope('pooling2'):
        net=tf.nn.max_pool(value=net,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    # 全连接
    with tf.variable_scope('fc1'):
        net=tf.reshape(net,shape=[-1,7*7*50])
        w=tf.get_variable(name='w',shape=[7*7*50,500],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        b=tf.get_variable(name='b',shape=[500],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        net = tf.nn.relu(tf.matmul(net,w)+b)
    # 全连接
    with tf.variable_scope('fc2'):
        w=tf.get_variable(name='w',shape=[500,10],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        b=tf.get_variable(name='b',shape=[10],initializer=tf.random_normal_initializer(mean=0,stddev=1))
        act=tf.nn.softmax(tf.matmul(net,w)+b)
        # 损失函数
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=act, labels=y))
    # 优化训练
    with tf.variable_scope('train'):
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    #评估函数
        pred = tf.equal(tf.argmax(act, axis=1), tf.argmax(y, axis=1))
    #正确率（True转换为1，False转换为0）
        acc = tf.reduce_mean(tf.cast(pred, tf.float32))

    init_op=tf.global_variables_initializer()

    with tf.Session() as sse:
        sse.run(init_op)
        for step in range(10):
            _,loss_r,acc_r=sse.run([train,cost,acc],feed_dict={x:train_img,y:train_label})
            print('第{}次迭代的损失值为：{},准确率为:{}'.format(step,loss_r,acc_r))