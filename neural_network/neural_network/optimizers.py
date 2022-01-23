import numpy as np


class SGD:
    def __init__(self, learning_rate: float = 0.01, momentum: int = 0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_updated = None

    def update(self, w: np.ndarray, grad_wrt_w):
        if self.weight_updated is None:
            self.weight_updated = np.zeros(np.shape(w))
        self.weight_updated = (
            self.momentum * self.weight_updated +
            (1 - self.momentum) * grad_wrt_w
        )
        return w - self.learning_rate * self.weight_updated


class Adam:
    def __init__(self, lr: float = 0.001, b1=0.9, b2=0.999):
        self.lr = lr
        self.eps = 1e-8
        self.b1 = b1
        self.b2 = b2
        self.m = None
        self.v = None

    def update(self, w, grad_wrt_w):
        if self.m is None or self.v is None:
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))
        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)
        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)
        self.delta_weight_updated = self.lr * \
            m_hat / (np.sqrt(v_hat) + self.eps)
        return w - self.delta_weight_updated
