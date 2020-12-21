from matplotlib import pyplot as plt
from colors import print_signs
from processing_data import load_train, load_test
from keras.models import Sequential, load_model
from keras.layers import Dense


name_model = 'BPNN_Model2/with_keras_model.h5'


def compile_model():
    # Load dataset
    X, Y = load_train()

    # Create model
    model = Sequential()
    model.add(Dense(512, input_dim=23, activation='relu'))
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


def evaluate_model():
    x_test, y_test = load_test()
    model = load_model(name_model)
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Loss - Accuracy -->: ", score)

    y_new = model.predict(x_test)

    for i in range(len(x_test)):
        print_signs(y_new[i], y_test[i])
