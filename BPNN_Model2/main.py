import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from colors import print_signs
from keras.models import Sequential, load_model
from keras.layers import Dense


name_model = 'keras_model.h5'


def load_dataset(name):
    df = pd.read_excel(name)
    return df


def pre_processing():
    df = load_dataset(r'dataset.xls').replace(r'^\s*$', np.nan, regex=True).dropna()
    df['ACCIDENTE'] = df['ACCIDENTE'].replace(to_replace=['No', 'Yes'], value=[0, 1])

    aux = df['ACCIDENTE']
    df.drop(columns=['FECHA_HORA', 'ACCIDENTE'], inplace=True)
    df = pd.get_dummies(df, columns=['TIPO_PRECIPITACION', 'INTENSIDAD_PRECIPITACION', 'ESTADO_CARRETERA'])
    df['ACCIDENTE'] = aux

    df = df.astype(float)

    dfX_test = df.sample(frac=0.05)
    df.drop(dfX_test.index[:], inplace=True)
    df.reset_index(drop=True, inplace=True)  # Creo que esto no hace falta
    dfX_test.reset_index(drop=True, inplace=True)  # Igual que esto tampoco
    dfX_test.to_excel("test.xls", index=False)

    x_train = df.iloc[:, 0:20].values
    y_train = df.iloc[:, 20].values

    return x_train, y_train


def compile_model():
    # Load dataset
    X, Y = pre_processing()

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
    history = model.fit(X, Y, epochs=50, validation_split=0.2, shuffle=True, batch_size=40)

    model.save(name_model)

    # plot metrics
    fig, (acc, loss) = plt.subplots(1, 2)
    fig.suptitle('Metrics')

    # Accuracy
    acc.plot(history.history['accuracy'])
    acc.plot(history.history['val_accuracy'])
    acc.set_title('Model accuracy')
    acc.set_ylabel('accuracy')
    acc.set_xlabel('epochs')
    acc.legend(['training', 'validation'], loc='lower right')
    acc.set_ylim([0, 1])

    # Loss
    loss.plot(history.history['loss'])
    loss.plot(history.history['val_loss'])
    loss.set_title('Model loss')
    loss.set_ylabel('loss')
    loss.set_xlabel('epochs')
    loss.legend(['training', 'validation'], loc='upper right')


def load_test():
    dfX_test = load_dataset(r'test.xls')
    y_test = dfX_test.iloc[:, 20].values
    x_test = dfX_test.iloc[:, 0:20].values

    return x_test, y_test


def evaluate_model():
    x_test, y_test = load_test()
    model = load_model(name_model)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Loss - Accuracy -->: ", score)

    y_new = model.predict(x_test)

    for i in range(len(x_test)):
        print_signs(y_new[i], y_test[i])
