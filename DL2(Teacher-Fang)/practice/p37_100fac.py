class MyBigInt:
    def __init__(self, n=0):
        self.digits = []
        while n > 0:
            self.digits.append(n % 10)
            n //= 10

    def __repr__(self):
        return ''.join([str(e) for e in reversed(self.digits)])

    def __str__(self):
        return self.__repr__()

    def __mul__(self, other):
        more = 0
        for i, digit in enumerate(self.digits):
            value = digit * other + more
            more = value // 10
            value %= 10
            self.digits[i] = value
        while more > 0:
            self.digits.append(more % 10)
            more //= 10
        return self

    def __rmul__(self, other):
        return self.__mul__(other)

def factorial(n):
    #result = 1
    result = MyBigInt(1)
    for i in range(2, n+1):
        result *= i
        print('%d != %s' % (i, result))

    return result


if __name__ == '__main__':
    print(factorial(100))