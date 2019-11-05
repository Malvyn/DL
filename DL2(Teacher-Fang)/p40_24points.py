
import numpy as np


def get_random_numbers(num, minimum, maximum):
    return np.random.randint(minimum, maximum, [num])


def get_values(numbers):
    """
    :param numbers: a list of the numbers which is great than zero
    :return: a list of the tuples of (value, exp) that the numbers can compose with addition(+), subtraction(-),
    multiplication(*) and/or division(/), as well as any number of paranthesis.
    """
    if len(numbers) == 1:
        return [(numbers[0], str(numbers[0]))]

    result = []
    for i in range(1, len(numbers)):
        for left_params in get_params_list(numbers, i):
            right_params = get_right_params(numbers, left_params)
            left = get_values(left_params)
            right = get_values(right_params)

            for left_value, left_exp in left:
                for right_value, right_exp in right:
                    result.append((left_value + right_value, '(' + left_exp + ' + ' + right_exp + ')'))
                    result.append((left_value - right_value, '(' + left_exp + ' - ' + right_exp + ')'))
                    result.append((right_value - left_value, '(' + right_exp + ' - ' + left_exp + ')'))
                    result.append((left_value * right_value, left_exp + ' * ' + right_exp))
                    if right_value != 0:
                        result.append((left_value / right_value, left_exp + ' / ' + right_exp))
                    if left_value != 0:
                        result.append((right_value / left_value, right_exp + ' / ' + left_exp))

    return result


def get_right_params(numbers, left_params):
    return [e for e in numbers if e not in left_params]


def get_params_list(numbers, num):
    if num == 0:
        yield []
    elif len(numbers) == num:
        yield [e for e in numbers]
    else:
        first = numbers[0]
        rest = numbers[1:]
        for params in get_params_list(rest, num-1):
            yield [first] + params

        if num <= len(rest):
            for params in get_params_list(rest, num):
                yield params


if __name__ == '__main__':
    numbers = get_random_numbers(4, 1, 20)
    print(numbers)

    for value, exp in get_values([2, 3, 7]):
        print(value, ',', exp)
    print('-' * 200)


    for value, exp in get_values(numbers):
        if value == 24:
            print(exp, '=', 24)




