


def hanoi(n, a, b, c):
    if n == 1:
        print('%s == %d ==> %s' % (a, n, c))

    else:
        hanoi(n-1, a, c, b)
        print('%s == %d ==> %s' % (a, n, c))
        hanoi(n-1, b, a, c)


def railway(ins, mid=0, outs=0):
    if ins == 0:
        return 1

    result = railway(ins-1, mid+1, outs)
    if mid > 0:
        result += railway(ins, mid-1, outs+1)

    return result


def rabits(n):
    if n < 2:
        return 1
    return rabits(n-1) + rabits(n-2)


def comb(m, n):
    if n == 0:
        return 1
    elif n == m:
        return 1

    return comb(m-1, n-1) + comb(m-1, n)


def test_comb(m, n, expect):
    c = comb(m, n)
    print('C(%d, %d) = %d, expect:%d, %s' % (m, n, c, expect, c==expect))


def match(source, pattern):
    if len(pattern) == 0:
        return len(source) == 0

    first = pattern[0]
    if first == '?':
        # return match(source, pattern[1:]) or (len(source) > 0 and match(source[1:], pattern[1:]))
        return len(source) > 0 and match(source[1:], pattern[1:])
    if first != '*':
        return len(source) > 0 and source[0] == first and match(source[1:], pattern[1:])

    return match(source, pattern[1:]) or len(source) > 0 and match(source[1:], pattern)


def test_match(source, pattern, expect):
    m = match(source, pattern)
    print('%s, %s, %s, expect: %s, %s' % (source, pattern, m, expect, m==expect))


if __name__ == '__main__':
    hanoi(4, 'A', 'B', 'C')

    for n in range(8):
        print('%d: %d' % (n, railway(n)))

    for n in range(8):
        print('%d: %d' % (n, rabits(n)))

    test_comb(5, 2, 10)
    test_comb(5, 3, 10)
    test_comb(6, 2, 15)
    test_comb(6, 3, 20)

    test_match('abaaab', 'a*b', True)
    test_match('abbaabb', 'a*a', False)
    test_match('bbbbaaaaa', '*a', True)
    test_match('bbbbaaaaa', '*ab', False)
    test_match('ababaabbaab', 'a?a*', True)
    test_match('ab', 'a*b', True)
    test_match('ab', 'a?b', False)

