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


def manhaDistance(vec1, vec2) -> float:
    return minkDistance(vec1, vec2, 1)


def chebyDistance(vector1, vector2) -> float:
    distance = 0.0
    if len(vector1) == len(vector2):
        distance = max(
            map(lambda vec1, vec2: abs(vec1 - vec2), vector1, vector2))
    return distance


def _mahaDistance(vector1, vector2, cov) -> float:
    npVec1 = np.array([vector1])
    npVec2 = np.array([vector2])

    minus = npVec1 - npVec2

    invCov = np.linalg.inv(cov)

    distance = np.sqrt(np.dot(np.dot(minus, invCov), minus.T))
    return distance


def mahaDistance(vector1, vector2, vectors) -> float:
    vecs = np.array(vectors)
    vecs = vecs.transpose()
    cov = np.cov(vecs)
    return _mahaDistance(vector1, vector2, cov)


def cosineDistance(vector1, vector2) -> float:
    distance = 0.0

    if len(vector1) == len(vector2):
        x = np.dot(vector1, vector2)

        def squareSum(vector):
            _sum = sum(map(lambda vec1: pow(vec1, 2), vector))
            return _sum

        y = pow(squareSum(vector1) * squareSum(vector2), 0.5)
        distance = x / y

    return distance


def calDistances(vectors, calculator=euclDistance):
    distances = []
    if callable(calculator):
        for vec in vectors:
            vec_diss = []
            for other_vec in vectors:
                vec_diss.append(calculator(vec, other_vec))
            distances.append(vec_diss)
    return distances


def pcc(vector1, vector2):
    r = 0

    if len(vector1) == len(vector2):
        mean1 = np.mean(vector1)
        mean2 = np.mean(vector2)
        x = sum(
            map(lambda vec1, vec2: (vec1 - mean1) * (vec2 - mean2), vector1,
                vector2)) / len(vector1)

        y = pow(np.var(vector1) * np.var(vector2), 0.5)

        r = x / y

    return r


x = 0

# x = mahaDistance([3, 4], [5, 6], [[3, 4], [5, 6], [2, 2], [8, 4]])
# x = cosineDistance([1, 2, 3], [2, 4, 6])
# x = calDistances([[3, 4], [5, 6], [2, 2], [8, 4]])
x = pcc([1, 2, 3, 4], [3, 8, 7, 6])
print(x)
