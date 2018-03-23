#!/usr/local/bin/python3

import numpy as np
# import scipy as sp


def minkDistance(vector1, vector2, dimenstion=2):
    distance = 0.0
    if len(vector1) == len(vector2):
        dis = sum(
            map(lambda vec1, vec2: pow(vec1 - vec2, dimenstion), vector1,
                vector2))
        distance = pow(dis, 1 / dimenstion)
    return distance


def euclDistance(vec1, vec2):
    return minkDistance(vec1, vec2, 2)


def manhaDistance(vec1, vec2):
    return minkDistance(vec1, vec2, 1)


def chebyDistance(vector1, vector2):
    distance = 0.0
    if len(vector1) == len(vector2):
        distance = max(
            map(lambda vec1, vec2: abs(vec1 - vec2), vector1, vector2))
    return distance


def _mahaDistance(vector1, vector2, cov):
    npVec1 = np.array([vector1])
    npVec2 = np.array([vector2])

    minus = npVec1 - npVec2

    invCov = np.linalg.inv(cov)

    distance = np.sqrt(np.dot(np.dot(minus, invCov), minus.T))
    return distance


def mahaDistance(vector1, vector2, vectors):
    vecs = np.array(vectors)
    vecs = vecs.transpose()
    cov = np.cov(vecs)
    return _mahaDistance(vector1, vector2, cov)


def cosineDistance(vector1, vector2):
    distance = 0.0

    if len(vector1) == len(vector2):
        x = np.dot(vector1, vector2)

        lengthFunc = lambda vector:pow(sum(map(lambda vec1: pow(vec1, 2), vector)), 0.5)

        y = lengthFunc(vector1) * lengthFunc(vector2)
        print(x, y)
        distance = x / y

    return distance


x = mahaDistance([3, 4], [5, 6], [[3, 4], [5, 6], [2, 2], [8, 4]])
print(x)


x = cosineDistance([1, 4], [3, 2])
print(x)
