

def set_queens(locations, nums):
    line_index = len(locations)
    if line_index == nums:
        return True

    for x in range(nums):
        if ok(x, line_index, locations):
            locations.append(x)
            if set_queens(locations, nums):
                return True
            del locations[-1]
    return False


def ok(x, line_index, locations):
    if x in locations:
        return False

    for i, xi in enumerate(locations):
        #  xi - x == i - line_index
        if line_index - i == xi - x or xi - x == i - line_index:
            return False
    return True


if __name__ == '__main__':
    locations = []
    if set_queens(locations, 8):
        for x in locations:
            print('- ' * x, end='')
            print('Q ', end='')
            print('- ' * (8 - 1 - x))