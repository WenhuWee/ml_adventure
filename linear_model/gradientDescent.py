#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def gradientDescent(x, y, lr=0.1, updateBatch=1):
    npX = x
    npY = y
    w = np.ones(npX.ndim)
    b = 0

    sumDeltaW = np.zeros(2)
    sumDeltaB = 0

    for index, x_ in enumerate(npX):

        print()
        print(index)
        print(w)
        print(b)

        y_ = npY[index]

        # print(y_)

        sumDeltaB = np.dot(x_, w) - y_
        sumDeltaW = x_ * (np.dot(x_, w) + b - y_)

        if (index + 1) % updateBatch == 0:

            print(index)

            w -= lr * sumDeltaW / updateBatch
            sumDeltaW = np.zeros(2)

            b -= lr * sumDeltaB / updateBatch
            sumDeltaB = 0

    return (w, b)


np.random.seed(123)
x = 100 * np.random.rand(5, 2)
w = np.array([3, 2])
y = 5 + np.dot(x, w) + np.random.randn(1) * 10

(w_p, b_p) = gradientDescent(x, y)

print(w_p)
print(b_p)

# x_test = 100 * np.random.rand(50, 2)
# y_p = b_p + np.dot(x_test, w_p)

# fig = plt.figure(figsize=(8, 6))
# plt.scatter(x, y)
# plt.scatter(x_test, y_p)
# plt.title("Dataset")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
