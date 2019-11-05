import os
import numpy as np
import tensorflow as tf

# 定义外部传入的参数
tf.app.flags.DEFINE_bool(flag_name="is_train",
                         default_value=True,
                         docstring="给定是否是训练操作，True表示训练，False表示预测！！")
tf.app.flags.DEFINE_float(flag_name="learning_rate",
                          default_value=0.001,
                          docstring="给定模型的学习率，默认0.001")
FLAGS = tf.app.flags.FLAGS


# 读取数据
def load_data(file_path):
    with open(file_path, 'r') as reader:
        data = reader.readlines()
    return data


# 单词转索引
def create_lookup_table(data):
    # 单词去重
    words = list(set(data))
    vocab_to_index = {word: idx for idx, word in enumerate(words)}
    index_to_vocab = {vocab_to_index[word]: word for word in words}
    return vocab_to_index, index_to_vocab


# 创建模型
def create_model(num_units=10, number_time_steps=3, vocab_size=120):
    with tf.variable_scope("Net", initializer=tf.random_normal_initializer(0.0, 0.1)):
        with tf.variable_scope("Input"):
            _x = tf.placeholder(tf.int32, shape=[None, 3])
            # 将输入的索引的类型修改为Float类型（tf.nn.static_bidirectional_rnn要求）
            _x = tf.cast(_x, tf.float32)
            _y = tf.placeholder(tf.int32, shape=[None])
            input_x = tf.split(_x, num_or_size_splits=number_time_steps, axis=1)
        with tf.variable_scope('rnn'):
            # a. 定义cell
            cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
            cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
            # 定义双向静态网络
            outputs, _, _ = tf.nn.static_bidirectional_rnn(cell_fw=cell_fw, cell_bw=cell_bw,
                                                           inputs=input_x, dtype=tf.float32)
        with tf.variable_scope('FC'):
            output = outputs[-1]
            # b. 初始化全连接的参数
            w = tf.get_variable(name='w', shape=[2 * num_units, vocab_size])
            b = tf.get_variable(name='b', shape=[vocab_size])
            logits = tf.add(tf.matmul(output, w), b)
        with tf.variable_scope("Predict"):
            y_predict = tf.argmax(logits, axis=1)
    return _x, _y, logits, y_predict

# 损失函数
def create_loss(logits, labels):
    with tf.variable_scope("loss"):
        labels = tf.reshape(labels, shape=[-1])
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    return loss

# 获取优化器及训练器
def create_optimizer(loss, learning_rate=0.001, global_step=None):
    with tf.variable_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
    return train_op

# 对数据进行处理
def create_X_and_Y(data, vocab_to_index, number_time_steps=3):
    X = []
    Y = []
    for content in data:
        words = content.strip().split(" ")
        words_number = len(words)
        offset = 0
        while offset < words_number - number_time_steps:
            tmp_x = words[offset:offset + number_time_steps]
            tmp_y = words[offset + number_time_steps]
            X.append([vocab_to_index[tx] for tx in tmp_x])
            Y.append(vocab_to_index[tmp_y])
            offset += 1
    X = np.asarray(X).reshape((-1, number_time_steps))
    Y = np.asarray(Y).reshape(-1)
    return X, Y


def train(file_path, max_epoch=10000, batch_size=100, num_units=10):
    # 1:执行图的创建
    # 加载数据
    data = load_data(file_path=file_path)
    # 将数据放在List
    text = []
    for line in data:
        line = line.strip()
        for word in line.split(" "):
            text.append(word)
    # 单词与索引的转换
    vocab_to_index, index_to_vocab = create_lookup_table(text)
    # 网络构建
    _x, _y, logits, y_predict = create_model(num_units=num_units,
                                             number_time_steps=3,
                                             vocab_size=len(vocab_to_index))
    # 损失函数构建
    loss = create_loss(logits=logits, labels=_y)
    # 构建优化器
    train_op = create_optimizer(loss)
    # 2：执行图的运行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        X, Y = create_X_and_Y(data, vocab_to_index=vocab_to_index, number_time_steps=3)
        total_samples = np.shape(X)[0] #总行数
        total_batch = total_samples // batch_size
        times = 0
        random_index = np.random.permutation(total_samples)
        for epoch in range(max_epoch):
            start_idx = times * batch_size  # 假设batch_size=8 start_idx=0
            end_idx = start_idx + batch_size #end_idx=8
            idx = random_index[start_idx:end_idx]
            train_x = X[idx]
            train_y = Y[idx]
            # 构建数据输入对象
            feed_dict = {
                _x: train_x,
                _y: train_y
            }
            # 模型训练
            _, _loss = sess.run([train_op, loss], feed_dict=feed_dict)

            if epoch % 100 == 0:
                idx = np.random.randint(0, total_samples)
                sample_in = np.reshape(X[idx], newshape=(-1, 3))
                sample_out = sess.run(y_predict, feed_dict={_x: sample_in})
                print("Epoch:{}, Loss:{}".format(epoch, _loss))
                print("%s - [%s] vs [%s]" % (X[idx], Y[idx], sample_out[0]))
            times += 1
            if times == total_batch:
                times = 0
                # 和之前保持一致
                random_index = np.random.permutation(total_samples)


def main(_):
    if FLAGS.is_train:
        train("belling_the_cat.txt")
    else:
        print("开始进行模型验证、测试代码运行.....")


if __name__ == '__main__':
    # 默认情况下，直接调用当前py文件中的main函数
    tf.app.run()
