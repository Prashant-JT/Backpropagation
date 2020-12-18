import numpy as np
from matplotlib import pyplot as plt
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
    df['ACCIDENTE'] = df['ACCIDENTE'].replace(to_replace=['No', 'Yes'], value=[0, 1])
    df.drop(columns=['FECHA_HORA', 'TIPO_PRECIPITACION', 'INTENSIDAD_PRECIPITACION', 'ESTADO_CARRETERA'],
            inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    features = df.iloc[:, 0:9].values
    label = df.iloc[:, 9].values
    return features, label


def test0():
    # Load dataset
    X, y = load_dataset()

    # Create model
    model = Sequential()
    model.add(Dense(32, input_dim=9, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    # Compile and fit model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, y, epochs=20, validation_split=0.2, shuffle=True, batch_size=30)

    """
    # plot metrics
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    """

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # Evaluate model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))



