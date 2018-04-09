#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


def gradientDescent(x, y, lr=0.01, updateBatch=2):

    count = x.shape[0]

    npX = x
    npY = y
    w = np.ones(npX.shape[1])
    b = 0

    sumDeltaW = np.zeros(npX.shape[1])
    sumDeltaB = 0
    sumloss = 0

    cost = []

    for index, x_ in enumerate(npX):

        y_ = npY[index]

        sumloss += (np.dot(x_, w) + b - y_)**2

        sumDeltaB += np.dot(x_, w) + b - y_
        sumDeltaW += x_ * (np.dot(x_, w) + b - y_)

        if count > 100 and index % (count / 100) == 0:
            averageLoss = sumloss / 100
            print('loss:', averageLoss)
            cost.append(averageLoss)
            sumloss = 0

        if (index + 1) % updateBatch == 0:
            w -= lr * sumDeltaW / updateBatch
            sumDeltaW = np.zeros(npX.shape[1])

            b -= lr * sumDeltaB / updateBatch
            sumDeltaB = 0

    return (w, b)


count = 5000
des = 1
np.random.seed(123)

x = 10 * np.random.rand(count, des)
w = np.array([4])
y = np.dot(x, w) + np.random.randn(count) * w[0].mean() / 2

(w_p, b_p) = gradientDescent(x, y)
print(w_p, b_p)

# clf = linear_model.SGDRegressor(penalty='none')
# clf.fit(x, y)

x_test = 10 * np.random.rand(50, des)
y_p = b_p + np.dot(x_test, w_p)
#
# # y_p = clf.predict(x_test)
#
fig = plt.figure()
plt.scatter(x, y)
plt.scatter(x_test, y_p)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# ax = Axes3D(fig)
# ax.scatter(x[:,0], x[:,1], y)
# ax.scatter(x_test[:,0], x_test[:,1], y_p)
# plt.show()
