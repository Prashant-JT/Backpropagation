import numpy as np
import matplotlib as mpl
import pandas as pd


result = {
    "0-20": 'very good',
    "21-40": 'good',
    "41-50": 'ok',
    "51-75": 'bad',
    "76-100": 'very bad'
}


def load_dataset():
    df = pd.read_excel(r'dataset.xls')
    df.drop(columns='FECHA_HORA', inplace=True)
    features = df.iloc[:, 0:12].values
    label = df.iloc[:,12:].values
    return features, label


def test0():
    # Load dataset
    X_train, Y_train = load_dataset()

    """
    # Create and fit model
    bpnn = BackPropagation(0.01, 50, 25)
    bpnn.fit(X_train[:55000], Y_train[:55000], X_train[55000:], Y_train[55000:])
    bpnn.predict(X_train)
    """