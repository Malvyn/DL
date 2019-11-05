def m():
    for i in range(1000000):
        yield i



if __name__ == '__main__':
    # for v in m():
    #     print(v)

    temp = m()
    while True:
        try:
            v = temp.__next__()
            print(v)
        except StopIteration:
            break

