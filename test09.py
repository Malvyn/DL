import os
import tensorflow as tf

tf.app.flags.DEFINE_bool(flag_name="is_train",
                         default_value=True,
                         docstring="给定是否是训练操作，True表示训练，False表示预测！！")
tf.app.flags.DEFINE_float(flag_name="learning_rate",
                          default_value=0.01,
                          docstring="给定学习率 默认是0.01！！")

FLAGS = tf.app.flags.FLAGS


def main(_):
    if FLAGS.is_train:
        print("开始进行模型训练运行.....")
    else:
        print("开始进行模型验证、测试代码运行.....")


if __name__ == '__main__':
    tf.app.run()
