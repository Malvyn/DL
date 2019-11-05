import math


class Exp:
    def __add__(self, other):
        print('in __add__')
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __truediv__(self, other):
        return truediv(self, other)

    def __rtruediv__(self, other):
        return truediv(other, self)

    def __pow__(self, power, modulo=None):
        return my_pow(self, power)

    def __rpow__(self, other):
        return my_pow(other, self)

    def __neg__(self):
        return neg(self)

    def deriv(self, x):
        if type(x) == str:
            x = Variable(x)
        return None

    def __str__(self):
        return self.__repr__()


class Const(Exp):
    def __init__(self, value=0):
        self.value = value

    def deriv(self, x):
        if type(x) == str:
            x = Variable(x)
        return Const(0)

    def __repr__(self):
        return str(self.value)


class Variable(Exp):
    def __init__(self, name):
        self.name = name

    def deriv(self, x):
        if type(x) == str:
            x = Variable(x)
        return Const(1) if x.name == self.name else Const(0)

    def __repr__(self):
        return self.name


pi = Const(math.pi)
e  = Const(math.e)


class Binary(Exp):
    def __init__(self, op: str, left: Exp, right: Exp):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return '(%s %s %s)' % (self.left, self.op, self.right)


class Add(Binary):
    def __init__(self, left, right):
        super(Add, self).__init__('+', left, right)

    def deriv(self, x):
        return self.left.deriv(x) + self.right.deriv(x)


def add(left, right):
    left = to_exp(left)
    right = to_exp(right)
    return Add(left, right)


def to_exp(right):
    if not isinstance(right, Exp):
        right = Const(right)
    return right


def sub(left, right):
    left = to_exp(left)
    right = to_exp(right)
    return Binary('-', left, right)


class Mul(Binary):
    def __init__(self, left, right):
        super(Mul, self).__init__('*', left, right)

    def deriv(self, x):
        return self.left.deriv(x) * self.right + self.left * self.right.deriv(x)


def mul(left, right):
    left = to_exp(left)
    right = to_exp(right)
    return Mul(left, right)


def truediv(left, right):
    left = to_exp(left)
    right = to_exp(right)
    return Binary('/', left, right)


def my_pow(left, right):
    left = to_exp(left)
    right = to_exp(right)
    return Binary('**', left, right)


def neg(left):
    left = to_exp(left)
    return -left.value


def log(value: Exp, base=e):
    value = to_exp(value)
    base = to_exp(base)
    return Binary('log', value, base)


if __name__ == '__main__':
    e = Const(50)
    print(e)  #  e.__repr__()
    print(str(e))  # e.__str__()
    print(e + 200)  # ==> print(e.__add__(200))

    print(e * 200)

    print(e - 200)

    print(e / 200)

    print(e ** 3)
    print(e ** -2.123459)

    print(-e)

    print(log(e))

    print(200+e)  # e.__radd__(200)

    print(200 - e)

    x = Variable('x')
    print(x)

    e = 30 + x
    print(e)
    print(e.deriv(x))

    e = x * x
    print(e)
    print(e.deriv(x))
