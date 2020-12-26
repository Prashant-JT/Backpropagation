from colors import print_signs
from processing_data import load_train, load_test
from BPNN_Model1.backpropagation import BackPropagation
import numpy as np


def compile_model():
    x_train, y_train = load_train()

    y_train = y_train.reshape(len(y_train), 1)

    index_valid = np.random.choice(x_train.shape[0], 5000, replace=False)

    x_valid = x_train[11780:11800, :]
    y_valid = y_train[11780:11800]

    model = BackPropagation(0.01, 50, 123, True, 40)

    """
    x_train = np.array([[2.7810836, 2.550537003],
                       [1.465489372, 2.362125076],
                       [3.396561688, 4.400293529],
                       [1.38807019, 1.850220317],
                       [3.06407232, 3.005305973],
                       [7.627531214, 2.759262235],
                       [5.332441248, 2.088626775],
                       [6.922596716, 1.77106367],
                       [8.675418651, -0.242068655],
                       [7.673756466, 3.508563011]])

    y_train = np.array([[0],
                       [0],
                       [0],
                       [0],
                       [0],
                       [1],
                       [1],
                       [1],
                       [1],
                       [1]])
    """

    model.fit(x_train, y_train, x_valid, y_valid, 6, np.array([512, 256, 128, 64, 32, 16]))

    return model


def evaluate_model(model):
    x_test, y_test = load_test()

    y_new = model.predict(x_test)

    for i in range(len(x_test)):
        print_signs(y_new[i], y_test[i])



