import numpy as np


# 加载数据
def loadDataSet():
    w = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    b = [0.35, 0.65]
    l = [5, 10]
    return w, b, l


# 激活函数
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


w, b, l = loadDataSet()


def f1(w, b, l):
    h1 = sigmoid(w[0] * l[0] + w[1] * l[1] + b[0])
    h2 = sigmoid(w[2] * l[0] + w[3] * l[1] + b[0])
    h3 = sigmoid(w[4] * l[0] + w[5] * l[1] + b[0])
    o1 = sigmoid(w[6] * h1 + w[8] * h2 + w[10] * h3 + b[1])
    o2 = sigmoid(w[7] * h1 + w[9] * h2 + w[11] * h3 + b[1])
    c1 = 0.1  # 目标值
    c2 = 0.99
    t1 = -(c1 - o1) * o1 * (1 - o1)
    t2 = -(c2 - o2) * o2 * (1 - o2)
    w[6] = w[6] - 0.5 * (t1 * h1)
    w[8] = w[8] - 0.5 * (t1 * h2)
    w[10] = w[10] - 0.5 * (t1 * h3)
    w[7] = w[7] - 0.5 * (t2 * h1)
    w[9] = w[9] - 0.5 * (t2 * h2)
    w[11] = w[11] - 0.5 * (t2 * h3)

    w[0] = w[0] - 0.5 * (t1 * w[6] + t2 * w[7]) * h1 * (1 - h1) * l[0]
    w[1] = w[1] - 0.5 * (t1 * w[6] + t2 * w[7]) * h1 * (1 - h1) * l[1]
    w[2] = w[2] - 0.5 * (t1 * w[8] + t2 * w[9]) * h2 * (1 - h2) * l[0]
    w[3] = w[3] - 0.5 * (t1 * w[8] + t2 * w[9]) * h2 * (1 - h2) * l[1]
    w[4] = w[4] - 0.5 * (t1 * w[10] + t2 * w[11]) * h3 * (1 - h3) * l[0]
    w[5] = w[5] - 0.5 * (t1 * w[10] + t2 * w[11]) * h3 * (1 - h3) * l[1]

    return o1, o2, w

def t_test():
    w, b, l = loadDataSet()
    for i in range(1001):
        r1, r2, w = f1(w, b, l)
        print("第{}次迭代后，结果值为:({},{}),权重更新为:{}".format(i, r1, r2, w))


t_test()