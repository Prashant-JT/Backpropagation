from BPNN_Model2.with_keras import compile_model as compile_keras, evaluate_model as evaluate_keras
from BPNN_Model1.without_keras import compile_model as compile_no_keras, evaluate_model as evaluate_no_keras
from BPNN_Model1.without_keras import test_sum_dataset, predict_test_dataset


# Run model with Keras
def run_keras():
    compile_keras()
    evaluate_keras()


# Run model without Keras
def run_no_keras():
    model = compile_no_keras()
    evaluate_no_keras(model)

    """
    Para probar el dataset de suma, comentar las dos 
    llamadas anteriores, y descomentar las dos siguientes.
    """

    # model = test_sum_dataset()
    # predict_test_dataset(model)
