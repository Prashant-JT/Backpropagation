import numpy
from BPNN_Model1 import layer


class OutputLayer(layer.Layer):

    def init_w(self, p_random_seed=numpy.random.RandomState(None)):
        self.w = p_random_seed.normal(loc=0.0,
                                      scale=0.01,
                                      size=(1 + self.number_inputs_each_neuron, self.number_neurons))
        return self

    def activation(self, z):
        signal = numpy.clip(z, -500, 500)
        return 1.0 / (1.0 + numpy.exp(-signal))

    def predict(self, p_X):
        return self.activation(self._net_input(p_X))
