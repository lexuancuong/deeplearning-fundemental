from typing import List

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, layers: List[int], lr: float = 0.1):
        self.layers = layers  # i.e [2,2,1]
        self.lr = lr  # learning rate
        self.W = []  # weights
        self.B = []  # biases

        for i in range(0, len(self.layers) - 1):
            # generate random weights matrix between layer (i)-th and (i+1)-th
            weight = np.random.randn(self.layers[i], self.layers[i + 1])
            bias = np.zeros((self.layers[i + 1], 1))
            self.W.append(weight / self.layers[i])
            self.B.append(bias)

    def fit_partial(self, x, y):
        A = [x]

        # feedforward
        z = A[0]  # This means out start with x input
        for i in range(0, len(self.layers) - 1):
            z = sigmoid(np.dot(z, self.W[i] + self.B[i].T))
            A.append(z)

        y = y.reshape(-1, 1)
        dA = [-(y / A[-1]) - (1 - y) / (1 - A[-1])]
        dW = []
        dB = []
        # backpropagation
        for i in reversed(range(0, len(self.layers) - 1)):
            dw = np.dot(A[i].T, dA[-1] * sigmoid_derivative(A[i + 1]))
            db = np.sum(
                (dA[-1]) * sigmoid_derivative(A[i + 1]),
                axis=0,
            ).reshape(-1, 1)
            da = np.dot(
                dA[-1] * sigmoid_derivative(A[i + 1]),
                self.W[i].T,
            )
            dW.append(dw)
            dA.append(da)
            dB.append(db)

        dW = dW[::-1]
        dB = dB[::-1]

        # Gradient descent
        for i in range(0, len(self.layers) - 1):
            self.W[i] = self.W[i] - self.lr * dW[i]
            self.B[i] = self.B[i] - self.lr * dB[i]

    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print(f"Epoch {epoch}, loss {loss}")

    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + (self.B[i].T))
        return X

    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        return -(np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))


nn = NeuralNetwork(layers=[2, 2, 1])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
nn.fit(X, y, epochs=10, verbose=1)
print(nn.predict([1, 1]))
