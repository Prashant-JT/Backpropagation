import numpy as np
from BPNN_Model1 import inputlayer, hiddenlayer, outputlayer


class BackPropagation(object):
    """Class BackPropagation:

       Attributes:
         eta.- Learning rate
         number_iterations.-
         ramdon_state.- Random process seed
         input_layer_.-
         hidden_layers_.-
         output_layer_.-
         sse_while_fit_.-

       Methods: __init__(p_eta=0.01, p_iterations_number=50, p_ramdon_state=1) fit(p_X_training, p_Y_training,
       p_X_validation, p_Y_validation, p_number_hidden_layers=1, p_number_neurons_hidden_layers=numpy.array([1]))
       predict(p_x) .- Method to predict the output, y

    """

    def __init__(self, p_eta=0.01, p_number_iterations=50, p_random_state=None, p_shuffle=True, p_batch_size=40):
        self.eta = p_eta
        self.number_iterations = p_number_iterations
        self.random_seed = np.random.RandomState(p_random_state)
        self.shuffle = p_shuffle
        self.batch_size = p_batch_size

    def fit(self, p_X_training, p_Y_training, p_X_validation, p_Y_validation,
            p_number_hidden_layers=1, p_number_neurons_hidden_layers=np.array([1])):

        self.input_layer_ = inputlayer.InputLayer(p_X_training.shape[1])
        self.hidden_layers_ = []
        for v_layer in range(p_number_hidden_layers):
            if v_layer == 0:
                self.hidden_layers_.append(hiddenlayer.HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                                   self.input_layer_.number_neurons))
            else:
                self.hidden_layers_.append(hiddenlayer.HiddenLayer(p_number_neurons_hidden_layers[v_layer],
                                                                   p_number_neurons_hidden_layers[v_layer - 1]))
        self.output_layer_ = outputlayer.OutputLayer(1,  # p_Y_training.shape[1] no funciona
                                                     self.hidden_layers_[
                                                         self.hidden_layers_.__len__() - 1].number_neurons)

        self.input_layer_.init_w(self.random_seed)
        for v_hidden_layer in self.hidden_layers_:
            v_hidden_layer.init_w(self.random_seed)
        self.output_layer_.init_w(self.random_seed)

        for i in range(self.number_iterations):
            indices = np.arange(p_X_training.shape[0])

            a_out = 0

            if self.shuffle:
                self.random_seed.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.batch_size + 1, self.batch_size):
                batch_idx = indices[start_idx: start_idx + self.batch_size]

                new_inputs, z_out, a_out = self.forward(p_X_training[batch_idx])

            print("Epoch ", i+1, " of ", self.number_iterations, "-----------------")

        return self

    def forward(self, x_data):
        new_inputs = []
        z_i = self.input_layer_._net_input(x_data)
        a_i = self.relu(z_i)
        new_inputs.append(a_i)
        for v_hidden_layer in self.hidden_layers_:
            z_h = v_hidden_layer._net_input(new_inputs[-1])
            a_h = self.relu(z_h)
            new_inputs.append(a_h)
        z_out = self.output_layer_._net_input(new_inputs[-1])
        a_out = self.sigmoid(z_out)

        return new_inputs, z_out, a_out

    def sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def predict(self, p_X):
        v_Y_input_layer_ = self.input_layer_.predict(p_X)
        v_X_hidden_layer_ = v_Y_input_layer_
        for v_hiddenlayer in self.hidden_layers_:
            v_Y_hidden_layer_ = v_hiddenlayer.predict(v_X_hidden_layer_)
            v_X_hidden_layer_ = v_Y_hidden_layer_
        v_X_output_layer_ = v_Y_hidden_layer_
        v_Y_output_layer_ = self.output_layer_.predict(v_X_output_layer_)
        return v_Y_output_layer_
