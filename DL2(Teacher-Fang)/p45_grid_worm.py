

def grid_worm(width, height):
    return _grid(0, 0, width, height)


def _grid(x, y, xt, yt):
    if x == xt and y == yt:
        return 1

    total = 0
    if x < xt:
        total += _grid(x+1, y, xt, yt)
    if y < yt:
        total += _grid(x, y+1, xt, yt)
    return total


if __name__ == '__main__':
    width = int(input('width = ?'))
    height = int(input('height = ?'))

    print('total = ', grid_worm(width, height))