import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import do_one_hot_encoding


def _sqr(number: float) -> float:
    return number * number


def generate_data(
    lower: int, higher: int, number: int, dimension: int, radius: int = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.uniform(low=lower, high=higher, size=(number, dimension))
    Y = np.array([math.sqrt(_sqr(x[0]) + _sqr(x[1]))
                 <= radius and 1 or 0 for x in X])

    return X, do_one_hot_encoding(Y.astype("int"))


if __name__ == "__main__":
    X, Y = generate_data(lower=-50, higher=50, number=1000, dimension=2)
    fig, ax = plt.subplots()
    circle = plt.Circle((0, 0), 5, color="#ffeecc")
    ax.add_patch(circle)

    for y in np.unique(Y):
        indexs = np.where(Y == y)
        _X = np.asarray(*[X[index] for index in indexs])
        ax.scatter(_X[:, 0], _X[:, 1], label=y)
    ax.legend()

    # Draw a circle
    plt.axis("scaled")

    plt.show()
