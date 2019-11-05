
def peach(monkeys):
    result = 1
    while not is_dividable(result, monkeys):
        result += monkeys
    return result


def is_dividable(peaches, monkeys):
    for _ in range(monkeys):
        peaches -= 1
        if peaches % monkeys == 0:
            peaches = (monkeys-1) * peaches / monkeys
        else:
            return False
    return True


if __name__ == '__main__':
    print(peach(5))