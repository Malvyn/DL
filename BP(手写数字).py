import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    with tf.Graph().as_default():
        unit_size1 = 32
        with tf.variable_scope('network',initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)):
            with tf.variable_scope("InputLayer"):
                input_x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='image')
                input_y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='label')
            with tf.variable_scope("Hidden_Layer_1"):
                w = tf.get_variable(name='w', shape=[784, unit_size1])
                b = tf.get_variable(name='b', shape=[unit_size1])
                # 2. 计算
                n1 = tf.matmul(input_x, w) + b
                # 3. 激活
                o1 = tf.nn.sigmoid(n1)
            with tf.variable_scope("Output_Layer"):
                w = tf.get_variable(name='w', shape=[unit_size1, 10])
                b = tf.get_variable(name='b', shape=[10])
                # 2. 计算
                z = tf.matmul(o1, w) + b
                p = tf.nn.softmax(z)
                y_ = tf.argmax(p, axis=1)
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(-tf.log(tf.reduce_sum(input_y * p, axis=1)), name='loss')
            tf.summary.scalar('loss', loss)
        with tf.name_scope('train'):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            train_op = optimizer.minimize(loss=loss)
        with tf.name_scope('accuracy'):
            label_ = tf.argmax(input_y, axis=1)
            tmp = tf.cast(tf.equal(y_, label_), tf.float32)
            accuracy = tf.reduce_mean(tmp)
            tf.summary.scalar('acc', accuracy)
        with tf.Session() as sess:
            # 创建一个持久化对象
            saver = tf.train.Saver()
            # 变量的初始化操作
            sess.run(tf.global_variables_initializer())
            # 获取一个日志输出对象
            writer = tf.summary.FileWriter(logdir='./models/01/graph', graph=sess.graph)
            # 获取所有的summary输出操作
            summary = tf.summary.merge_all()
            # b. 训练数据的产生/获取
            mnist = input_data.read_data_sets(
                train_dir='datas/mnist',  # 给定本地磁盘的数据存储路径
                one_hot=True,  # 给定返回的数据中是否对Y做哑编码
                validation_size=5000  # 给定验证数据集的大小
            )
            x_train, y_train = mnist.train.images, mnist.train.labels
            x_test, y_test = mnist.test.images, mnist.test.labels
            # c. 模型训练
            for step in range(100):
                _, loss_, accuracy_, summary_ = sess.run([train_op, loss, accuracy, summary], feed_dict={
                    input_x: x_train,
                    input_y: y_train
                })
                print("第{}次训练后模型的损失函数为:{}, 准确率:{}".format(step, loss_, accuracy_))
            writer.add_summary(summary_, global_step=step)
            # 触发模型持久化
            save_path = './models/01/model/model.ckpt'
            dirpath = os.path.dirname(save_path)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            saver.save(sess, save_path=save_path)












































