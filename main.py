from using_keras.with_keras import compile_model as compile_keras, evaluate_model as evaluate_keras
from not_using_keras.without_keras import compile_model as compile_no_keras


# Run model with Keras
def run_keras():
    compile_keras()
    evaluate_keras()


# Run model without Keras
def run_no_keras():
    compile_no_keras()
    # evaluate_model()
