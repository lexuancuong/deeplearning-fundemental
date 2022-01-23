from __future__ import division, print_function

from typing import Callable

import numpy as np
import progressbar
from tabulate import tabulate
from terminaltables import AsciiTable

from utils import batch_iterator

bar_widgets = [
    "Training: ",
    progressbar.Percentage(),
    " ",
    progressbar.Bar(marker="=", left="[", right="]"),
    " ",
    progressbar.ETA(),
]


class NeuralNetwork:
    def __init__(
        self, optimizer: Callable, loss: Callable, validation_data: tuple = None
    ):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}

    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())
        if hasattr(layer, "initialize"):
            layer.initialize(optimizer=self.optimizer)
        self.layers.append(layer)

    def summary(
        self, accuracy: float = 0, loss: float = 0, training_time: float = 0
    ) -> None:
        layers_table_headers = ["Layer", "Parameters", "Output Shape"]
        layers_table_data = []
        total_parameters = 0
        for layer in self.layers:
            layer_name = layer.layer_name
            params = layer.parameters
            output_shape = layer.output_shape()
            layers_table_data.append(
                [layer_name, str(params), str(output_shape)])
            total_parameters += params

        result_table_headers = ["Total parameters",
                                "Accuracy", "Loss", "Training time"]
        result_table_data = [[total_parameters, accuracy, loss, training_time]]
        print(tabulate(layers_table_data, layers_table_headers, tablefmt="fancy_grid"))
        print(tabulate(result_table_data, result_table_headers, tablefmt="fancy_grid"))

    def predict(self, X):
        return self._forward_pass(X, is_training=False)

    def test_on_batch(self, X, y):
        y_pred = self._forward_pass(X, is_training=False)
        loss = np.mean(self.loss_function.cal_loss(y, y_pred))
        acc = self.loss_function.cal_acc(y, y_pred)

        return loss, acc

    def train_on_batch(self, X, y):
        y_pred = self._forward_pass(X)
        loss = np.mean(self.loss_function.cal_loss(y, y_pred))
        acc = self.loss_function.cal_acc(y, y_pred)
        loss_grad = self.loss_function.cal_gradient(y, y_pred)
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc

    def fit(self, X, y, n_epochs, batch_size):
        for _ in self.progressbar(range(n_epochs)):

            batch_error = []
            for X_batch, y_batch in batch_iterator(X, y, batch_size=batch_size):
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_error.append(loss)

            self.errors["training"].append(np.mean(batch_error))

            if self.val_set is not None:
                val_loss, _ = self.test_on_batch(
                    self.val_set["X"], self.val_set["y"])
                self.errors["validation"].append(val_loss)

        return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, X, is_training=True):
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, is_training)

        return layer_output

    def _backward_pass(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_pass(loss_grad)
