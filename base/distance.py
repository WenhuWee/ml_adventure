#!/usr/local/bin/python3


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


x = chebyDistance([1, 2], [3, 5])
print(x)
