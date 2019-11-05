


def m(a):
    a = a + 100


class A:
    def __init__(self):
        self.a = 999

def m2(a):
    #  a: the address of the A object
    a.a += 100

if __name__ == '__main__':
    # call by value
    b = 555
    m(b)
    print(b)

    # call by reference/address
    b2 = A()   # The address of the A object is stored in b2
    print(b2.a)
    m2(b2)   # call by reference
    print(b2.a)
