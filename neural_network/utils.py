from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


def batch_iterator(X, y=None, batch_size=256):
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i + batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def calculate_covariance_matrix(X, Y=None):
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(
        Y - Y.mean(axis=0)
    )
    return np.array(covariance_matrix, dtype=float)


def _transform(X, dim):
    covariance = calculate_covariance_matrix(X)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx][:dim]
    eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :dim]
    X_transformed = X.dot(eigenvectors)

    return X_transformed


def save_distribution(
    X,
    filepath: str,
    y=None,
    title=None,
    accuracy=None,
    legend_labels=None,
):
    X_transformed = _transform(X, dim=2)
    x1 = X_transformed[:, 0]
    x2 = X_transformed[:, 1]
    class_distr = []
    y = np.array(y).astype(int)
    _cmap = plt.get_cmap("viridis")
    colors = [_cmap(i) for i in np.linspace(0, 1, len(np.unique(y)))]
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_distr.append(plt.scatter(_x1, _x2, color=colors[i]))
    if not legend_labels is None:
        plt.legend(class_distr, legend_labels, loc=1)
    if accuracy:
        perc = 100 * accuracy
        plt.suptitle(title)
        title = f"Accuracy: {perc}%"
    plt.title(title)

    # Axis labels
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("scaled")
    plt.savefig(filepath)
    plt.clf()


def do_one_hot_encoding(x: np.ndarray, n_col: int = None) -> np.ndarray:
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def save_losses_chart(train_err: List, val_err: List, filepath: str) -> None:
    n = len(train_err)
    (training,) = plt.plot(range(n), train_err, label="Training Error")
    (validation,) = plt.plot(range(n), val_err, label="Validation Error")
    plt.legend(handles=[training, validation])
    plt.title("Losses on Training and Validation dataset")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.savefig(filepath)
    plt.clf()
