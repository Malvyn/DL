import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

tf.app.flags.DEFINE_float(flag_name="dropout_prob",
                          default_value=0.75,
                          docstring="给定模型的Dropout，默认0.75")

tf.app.flags.DEFINE_bool(flag_name="is_train",
                         default_value=True,
                         docstring="给定是否是训练操作，True表示训练，False表示预测！！")
tf.app.flags.DEFINE_integer(flag_name="batch_size",
                            default_value=16,
                            docstring="给定训练的时候每个批次的样本数目，默认为16.")
tf.app.flags.DEFINE_integer(flag_name="display_step",
                            default_value=10,
                            docstring="测试显示的间隔，默认为10")
tf.app.flags.DEFINE_float(flag_name="learning_rate",
                          default_value=0.001,
                          docstring="给定模型的学习率，默认0.001")

FLAGS = tf.app.flags.FLAGS


# 定义卷积操作
def conv2d(name, x, w, b, strides=1):
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)


# 定义池化操作
def max_pooling(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 创建模型
def create_model(x, weights, biases, dropout):
    # 输入层
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # 第一层卷积
    conv1 = conv2d("conv1", x, weights['wc1'], biases['bc1'])
    # 池化层
    pooling1 = max_pooling("pooling1", conv1, k=2)
    # 第二层卷积
    conv2 = conv2d("conv2", pooling1, weights['wc2'], biases['bc2'])
    pooling2 = max_pooling("pooling2", conv2, k=2)
    # 第三层卷积
    conv3 = conv2d("conv3", pooling2, weights['wc3'], biases['bc3'])
    # 第四层卷积
    conv4 = conv2d("conv4_layer", conv3, weights['wc4'], biases['bc4'])
    # 第五层卷积
    conv5 = conv2d("conv5_layer", conv4, weights['wc5'], biases['bc5'])
    pooling5 = max_pooling("pooling5", conv5, k=2)
    # 第六层 三维拉平到一维
    fc1 = tf.reshape(pooling5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.nn.dropout(fc1, keep_prob=dropout)
    fc1 = tf.nn.relu(tf.matmul(fc1, weights['wd1']) + biases['bd1'])

    # 第七层
    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.nn.relu(tf.matmul(fc2, weights['wd2']) + biases['bd2'])
    fc2 = tf.nn.dropout(fc2, keep_prob=dropout)
    pred = tf.matmul(fc2, weights['out']) + biases['out']
    return pred


# 创建损失函数
def create_loss(labels, pred):
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred))
    return loss


# 创建训练器
def create_train_op(loss, learning_rate=0.001):
    with tf.name_scope("train"):
        train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)
    return train_op


# 精度获取
def create_accuracy(y, pred):
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


# 训练模型
def train():
    # 1:执行图的创建
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    drop_prob = tf.placeholder(tf.float32)
    weights = {
        # 卷积核filter大小11*11 输入层为1个feature maps，输出层有96 feature maps
        'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
        # 卷积核filter大小5*5 输入层为192个feature maps，输出层有384 feature maps
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        'wd1': tf.Variable(tf.random_normal([4 * 4 * 256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        'out': tf.Variable(tf.random_normal([4096, 10]))
    }
    biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([10]))
    }
    # 1. 网络结构的构建
    pred = create_model(x, weights, biases, drop_prob)
    # 2. 构建损失函数
    cost = create_loss(y, pred)
    # 3. 构建优化器
    train_op = create_train_op(cost, learning_rate=0.001)
    # 4. 准确率
    accuracy = create_accuracy(y, pred)
    # 2：执行图的运行
    # 获取数据
    mnist = input_data.read_data_sets(
        train_dir='datas/mnist',  # 给定本地磁盘的数据存储路径
        one_hot=True,  # 给定返回的数据中是否对Y做哑编码
        validation_size=5000  # 给定验证数据集的大小
    )
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, drop_prob: FLAGS.dropout_prob})
            if step % FLAGS.display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, drop_prob: 1.})
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, drop_prob: 1.})
                print("Iter " + str(step * FLAGS.batch_size) + ", Minibatch Loss = " + "{:.6f}".format(
                    loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
            step += 1


# 模型预测
def prediction():
    pass


def main(_):
    if FLAGS.is_train:
        train()
    else:
        print("开始进行模型验证、测试代码运行.....")


if __name__ == '__main__':
    # 默认情况下，直接调用当前py文件中的main函数
    tf.app.run()
