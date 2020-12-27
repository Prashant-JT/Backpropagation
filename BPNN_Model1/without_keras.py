from random import random
from colors import print_signs
from processing_data import load_train, load_test
from BPNN_Model1.backpropagation import BackPropagation
import numpy as np


def compile_model():
    x_train, y_train = load_train()

    y_train = y_train.reshape(len(y_train), 1)

    index_valid = np.random.choice(x_train.shape[0], 5000, replace=False)

    x_valid = x_train[index_valid, :]
    y_valid = y_train[index_valid]

    model = BackPropagation(0.001, 50, 1, True, 40)

    model.fit(x_train, y_train, x_valid, y_valid, 6, np.array([512, 256, 128, 64, 32, 16]))

    return model


def evaluate_model(model):
    x_test, y_test = load_test()

    y_new = model.predict(x_test)

    for i in range(len(x_test)):
        print_signs(y_new[i], y_test[i])


def test_sum_dataset():
    items = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    x_train = items
    y_train = targets

    index_valid = np.random.choice(x_train.shape[0], 50, replace=False)

    x_valid = x_train[index_valid, :]
    y_valid = y_train[index_valid]

    model = BackPropagation(0.2, 100, 1, True, 1)

    model.fit(x_train, y_train, x_valid, y_valid, 1, np.array([10]))

    return model


def predict_test_dataset(model):
    input_ = np.array([[0.3, 0.1], [0.2, 0.7]])
    target = np.array([[0.4], [0.9]])

    y_predict = model.predict(input_)

    for a, b in zip(y_predict, target):
        print("Predicted: ", a, " Real: ", b)



