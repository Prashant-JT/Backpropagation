import numpy as np
import matplotlib as mpl
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


result = {
    "0-20": 'very good',
    "21-40": 'good',
    "41-50": 'ok',
    "51-75": 'bad',
    "76-100": 'very bad'
}


def load_dataset():
    df = pd.read_excel(r'dataset.xls')
    df.drop(columns=['FECHA_HORA', 'TIPO_PRECIPITACION','INTENSIDAD_PRECIPITACION', 'ESTADO_CARRETERA'], inplace=True)
    features = df.iloc[:, 0:9].values
    label = df.iloc[:, 9].values
    return features, label


def test0():
    # Load dataset
    X, y = load_dataset()

    #X = np.asarray(X).astype(int)
    #print(X.shape)
    #print(X[0].astype(float))

    """
    # Create model
    model = Sequential()
    model.add(Dense(24, input_dim=12, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    # Compile and fit model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy, loss'])
    model.fit(X, y, epochs=20, batch_size=10)

    # Evaluate model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    """