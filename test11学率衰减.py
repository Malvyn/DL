import tensorflow as tf
import matplotlib.pyplot as plt

def f1():
    num_epoch = tf.Variable(0, name='epoch', trainable=False)
    assign_op = tf.assign_add(num_epoch, 1)
    boundaries = [10, 30, 70] #分界
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    with tf.control_dependencies([assign_op]):
        learning_rate = tf.train.piecewise_constant(
            x=num_epoch, boundaries=boundaries, values=learning_rates
        )
    N = 100
    y = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(N):
            lr = sess.run(learning_rate)
            y.append(lr)
    plt.plot(y, 'r-')
    plt.show()

def f2():
    num_epoch = tf.Variable(0, name='epoch', trainable=False)
    assign_op = tf.assign_add(num_epoch, 1)
    base_learning_rate = 0.1
    decay_steps = 10
    with tf.control_dependencies([assign_op]):
        # decayed_learning_rate = learning_rate * exp(-decay_rate * global_step / decay_steps)
        # learning_rate：初始学习率
        # global_step：当前的训练批次
        # decay_steps：衰减周期(每隔多少批次衰减一次)
        # decay_rate: 衰减率系数
        # staircase:是否做阶梯型的衰减还是连续衰减，默认False为连续衰减
        learning_rate1 = tf.train.natural_exp_decay(
            learning_rate=base_learning_rate,
            global_step=num_epoch,
            decay_steps=decay_steps,
            decay_rate=0.9,
            staircase=False
        )
        learning_rate2 = tf.train.natural_exp_decay(
            learning_rate=base_learning_rate,
            global_step=num_epoch,
            decay_steps=decay_steps,
            decay_rate=0.9,
            staircase=True
        )
    N = 100
    y1 = []
    y2 = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(N):
            lr1, lr2 = sess.run([learning_rate1, learning_rate2])
            y1.append(lr1)
            y2.append(lr2)
    plt.plot(y1, 'r-')
    plt.plot(y2, 'g-')
    plt.show()


f2()











