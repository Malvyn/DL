# coding=utf-8


#  solve argmin(y, x1, x2)


def y(x1, x2):
    return (x1 - 3)**2 + (x2 - 2.5)**2


def y_x1(x1, x2):
    return 2*(x1 - 3)


def y_x2(x1, x2):
    return 2*(x2 - 2.5)


x1, x2 = 1.0, 1.0
lr = 0.001

for _ in range(20000):
    dx1 = - y_x1(x1, x2) * lr
    dx2 = - y_x2(x1, x2) * lr

    x1 += dx1
    x2 += dx2

print(x1, x2)

