from __future__ import division

import numpy as np

from utils import accuracy_score


class Loss(object):
    def cal_loss(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return NotImplementedError()

    def cal_gradient(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError()

    def cal_acc(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))


class SquareLoss(Loss):
    def __init__(self):
        ...

    def cal_loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def cal_gradient(self, y, y_pred):
        return -(y - y_pred)


class CrossEntropy(Loss):
    def __init__(self):
        ...

    def cal_loss(self, y, p):
        p = np.clip(p, 1e-13, 1 - 1e-13)
        return -y * np.log(p) - (1 - y) * np.log(1 - p)

    def cal_gradient(self, y, p):
        p = np.clip(p, 1e-13, 1 - 1e-13)
        return -(y / p) + (1 - y) / (1 - p)
