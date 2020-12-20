from processing_data import load_train, load_test
from BPNN_Model1.backpropagation import BackPropagation
import numpy as np


def compile_model():
    x_train, y_train = load_train()

    index_valid = np.random.choice(x_train.shape[0], 5000, replace=False)

    x_valid = x_train[index_valid, :]
    y_valid = y_train[index_valid]

    model = BackPropagation(0.01, 10, 1)

    model.fit(x_train, y_train, x_valid, y_valid, 5, np.array([256, 128, 64, 32, 16]))

