import numpy as np


def isNeighbour(x, y, horizon):
    l2 = 0
    for i in range(len(x)):
        l2 += (x[i] - y[i]) * (x[i] - y[i])
    l2 = np.sqrt(l2)
    out = 0
    if(l2 < horizon):
        out = 1
    return out


def l2(y1, y2):
    l2 = 0
    for i in range(len(y1)):
        l2 += (y1[i] - y2[i]) * (y1[i] - y2[i])
    l2 = np.sqrt(l2)
    return l2


def l2_sqr(y1, y2):
    l2 = 0
    for i in range(len(y1)):
        l2 += (y1[i] - y2[i]) * (y1[i] - y2[i])
    return l2


def l2norm(x):
    l2 = 0
    for i in range(len(x)):
        l2 += x[i] * x[i]
    l2 = np.sqrt(l2)
    return l2
