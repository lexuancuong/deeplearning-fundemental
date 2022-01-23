from __future__ import print_function

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import progressbar

from data import generate_data
from layers import Activation, Dense, Dropout
from loss_functions import CrossEntropy, SquareLoss
from model import NeuralNetwork
from optimizers import SGD, Adam
from utils import save_distribution, save_losses_chart

bar_widgets = [
    "Training: ",
    progressbar.Percentage(),
    " ",
    progressbar.Bar(marker="-", left="[", right="]"),
    " ",
    progressbar.ETA(),
]

LOWER = -3.13
HIGHER = 3.13


def process(
    layers: list,
    loss_function: object,
    optimizer: object,
    epoches: int,
    folder: str,
    dropout: float = 0,
) -> None:
    if not os.path.exists(folder):
        os.makedirs(folder)
    X_train, y_train = generate_data(
        lower=LOWER, higher=HIGHER, number=100000, dimension=2
    )
    X_validate, y_validate = generate_data(
        lower=LOWER, higher=HIGHER, number=20000, dimension=2
    )
    X_test, y_test = generate_data(
        lower=LOWER, higher=HIGHER, number=20000, dimension=2
    )
    save_distribution(
        X=X_train,
        y=np.argmax(y_train, axis=1),
        filepath=os.path.join(folder, "distribution_of_training_dataset.png"),
        title="The distribution of the training dataset",
        legend_labels=range(2),
    )
    nn = NeuralNetwork(
        optimizer=optimizer,
        loss=loss_function,
        validation_data=(X_validate, y_validate),
    )
    for layer in layers:
        nn.add(Dense(layer, input_shape=(X_train.shape[1],)))
        nn.add(Activation("sigmoid"))
        if dropout:
            nn.add(Dropout(dropout))
    nn.add(Dense(2))
    nn.add(Activation("softmax"))

    start = time.perf_counter()
    train_err, val_err = nn.fit(
        X_train, y_train, n_epochs=epoches, batch_size=256)
    training_time = time.perf_counter() - start

    save_losses_chart(
        train_err=train_err,
        val_err=val_err,
        filepath=os.path.join(
            folder, "loss_on_training_and_validation_dataset.png"),
    )
    loss, accuracy = nn.test_on_batch(X_test, y_test)
    nn.summary(accuracy=accuracy, loss=loss, training_time=training_time)

    y_pred = np.argmax(nn.predict(X_test), axis=1)
    save_distribution(
        X=X_test,
        y=y_pred,
        filepath=os.path.join(
            folder, "distribution_of_predicted_result_on_testing_dataset.png"
        ),
        title="The distribution of the predicted result",
        accuracy=accuracy,
        legend_labels=range(10),
    )


if __name__ == "__main__":
    if not os.path.exists("result"):
        os.makedirs("result")

    # MLP with one hidden layer, 3 perceptrons, L2 loss, SGD optimizer, no dropout, 10 epoches
    process(
        [3],
        loss_function=SquareLoss,
        optimizer=SGD(),
        epoches=10,
        folder="result/requirement_1",
    )

    # MLP with one hidden layer, 3 perceptrons, L2 loss, SGD optimizer, no dropout, 500 epoches
    process(
        [3],
        loss_function=SquareLoss,
        optimizer=SGD(),
        epoches=500,
        folder="result/requirement_2",
    )

    # MLP with one hidden layer, 128 perceptrons, CE loss, Adam optimizer, no dropout, 100 epoches
    process(
        [128],
        loss_function=CrossEntropy,
        optimizer=Adam(),
        epoches=100,
        folder="result/requirement_3",
    )

    # MLP with three hidden layer, (32, 64, 32)  perceptrons, CE loss, Adam  optimizer, no dropout, 100 epoches
    process(
        layers=[32, 64, 32],
        loss_function=CrossEntropy,
        optimizer=Adam(),
        epoches=100,
        folder="result/requirement_4",
    )

    # MLP with three hidden layer, (32, 64, 32)  perceptrons, CE loss, Adam  optimizer, 20% dropout, 100 epoches
    process(
        layers=[32, 64, 32],
        loss_function=CrossEntropy,
        optimizer=Adam(),
        epoches=100,
        folder="result/requirement_5",
        dropout=0.2,
    )
