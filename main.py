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
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
    df['ACCIDENTE'] = df['ACCIDENTE'].replace(to_replace=['No', 'Yes'], value=[0, 1])

    aux = df['ACCIDENTE']
    df.drop(columns=['FECHA_HORA', 'ACCIDENTE'], inplace=True)
    df = pd.get_dummies(df, columns=['TIPO_PRECIPITACION', 'INTENSIDAD_PRECIPITACION', 'ESTADO_CARRETERA'])
    df['ACCIDENTE'] = aux

    df = df.astype(float)

    dfX_test = df.sample(frac=0.1)
    df.drop(dfX_test.index[:], inplace=True)

    dfX_test.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    y_test = dfX_test.iloc[:, 20].values
    dfX_test.drop(columns=['ACCIDENTE'], inplace=True)
    x_test = dfX_test.iloc[:, 0:20].values

    features = df.iloc[:, 0:20].values
    label = df.iloc[:, 20].values
    return features, label, x_test, y_test


def test0():
    # Load dataset
    X, y, x_test, y_test = load_dataset()

    # Create model
    model = Sequential()
    model.add(Dense(512, input_dim=20, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    # Compile and fit model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=50, validation_split=0.2, shuffle=True, batch_size=50)

    score = model.evaluate(x_test, y_test, verbose=1)
    print("Loss - Accuracy -->: ", score)

    # y_new = model.predict(x_test)

    """
    for i in range(len(x_test)):
        if np.round(y_new[i]) != y_test[i]:
            print("Original: ", y_test[i], " ---- Predicted: ", y_new[i], " -------------------- No coincide ---------")
        else:
            print("Original: ", y_test[i], " ---- Predicted: ", y_new[i])

    
    # plot metrics
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    

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
    """