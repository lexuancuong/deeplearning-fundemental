from __future__ import division, print_function

import copy
import math

import numpy as np

from activation_functions import get_activation_function


class Layer(object):
    def __init__(self):
        self.trainable = True

    @property
    def layer_name(self):
        return self.__class__.__name__

    @property
    def parameters(self):
        return 0

    def set_input_shape(self, shape):
        self.input_shape = shape

    def forward_pass(self, X: np.ndarray, is_training: bool):
        raise NotImplementedError()

    def backward_pass(self, accum_grad: np.ndarray):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):
    def __init__(self, n_units: int, input_shape: tuple = None):
        super().__init__()
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit,
                                   (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    @property
    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, is_training: bool = True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units,)


class Dropout(Layer):
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p
        self._mask = None
        self.input_shape = None
        self.n_units = None
        self.pass_through = True
        self.trainable = True

    def forward_pass(self, X: np.ndarray, is_training: bool = True):
        c = 1 - self.p
        if is_training:
            self._mask = np.random.uniform(size=X.shape) > self.p
            c = self._mask
        return X * c

    def backward_pass(self, accum_grad):
        return accum_grad * self._mask

    def output_shape(self):
        return self.input_shape


class Activation(Layer):
    def __init__(self, name):
        super().__init__()
        self.activation_name = name
        self.activation_func = get_activation_function(name)()

    @property
    def layer_name(self):
        return f"Activation {self.activation_func.__class__.__name__}"

    def forward_pass(self, X, is_training=True):
        self.layer_input = X
        return self.activation_func(self.layer_input)

    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape
