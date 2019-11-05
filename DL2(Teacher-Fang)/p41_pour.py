

def pour(capacity1, capacity2, goal, v1, v2, visited=[]):
    current = (v1, v2)
    if current in visited:
        return False

    visited.append(current)

    if v1 == goal or v2 == goal:
        return True

    if v1 > 0 and pour(capacity1, capacity2, goal, 0, v2, visited):
        return True

    if v2 > 0 and pour(capacity1, capacity2, goal, v1, 0, visited):
        return True

    if v1 < capacity1 and pour(capacity1, capacity2, goal, capacity1, v2, visited):
        return True

    if v2 < capacity2 and pour(capacity1, capacity2, goal, v1, capacity2, visited):
        return True

    if v1 + v2 <= capacity2 and pour(capacity1, capacity2, goal, 0, v1+v2, visited):
        return True

    if v1 + v2 <= capacity1 and pour(capacity1, capacity2, goal, v1+v2, 0, visited):
        return True

    if v1 + v2 > capacity2 and pour(capacity1, capacity2, goal, v1 + v2 - capacity2, capacity2, visited):
        return True

    if v1 + v2 > capacity1 and pour(capacity1, capacity2, goal, capacity1, v1 + v2 - capacity1, visited):
        return True

    del visited[-1]
    return False


def test_pour(capacity1, capacity2, goal):
    path = []
    if pour(capacity1, capacity2, goal, 0, 0, path):
        print(path)
    else:
        print('no solution')


if __name__ == '__main__':
    test_pour(5, 3, 4)
    test_pour(6, 4, 5)
