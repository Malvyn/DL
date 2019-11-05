# coding=utf-8


import tensorflow as tf


#  Use Gradient Descent to get sqrt(x)


def sqrt(a):
    def y(x):
        return x*x - a
    # y = lambda x: x*x - a

    def loss(x):
        return y(x) ** 2

    def deriv(x):
        return 4 * x * y(x)

    epoches = 1000
    lr = 0.001
    x = 1.0

    for _ in range(epoches):
        delta_x = - lr * deriv(x)
        x += delta_x

    return x


def sqrt3(a):
    def y(x):
        return x**3 - a

    def loss(x):
        return y(x) ** 2

    def deriv(x):
        return 2 * y(x) * 3 * x**2

    epoches = 1000
    lr = 0.001
    x = 1.0

    for _ in range(epoches):
        delta_x = - lr * deriv(x)
        x += delta_x

    return x


if __name__ == '__main__':
    for x in range(2, 10):
        print('sqrt(%s) = %s' % (x, sqrt(x)))
        print('sqrt3(%s) = %s' % (x, sqrt3(x)))
