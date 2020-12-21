from BPNN_Model2.with_keras import compile_model as compile_keras, evaluate_model as evaluate_keras
from BPNN_Model1.without_keras import compile_model as compile_no_keras


# Run model with Keras
def run_keras():
    compile_keras()
    evaluate_keras()


# Run model without Keras
def run_no_keras():
    compile_no_keras()
    # evaluate_model()
