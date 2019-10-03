import numpy as np


def train(x1, x2, w1, w2, bias):
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    tmp = np.sum(w*x) + bias
    if tmp <= 0:
        return 0
    else:
        return 1


def resolve(name, w1, w2, bias):
    print('Result of ' + name + ':')
    array = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for item in array:
        result = train(item[0], item[1], w1, w2, bias)
        print(str(item) + " -> " + str(result))


if __name__ == '__main__':
    resolve('AND', 0.5, 0.5, -0.7)
    resolve('OR', 0.5, 0.5, - 0.2)
    resolve('NAND', -0.5, -0.5, 0.7)
