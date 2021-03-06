"""PerceptronNetwork Code"""

from .perceptronlayer import PerceptronLayer
from .tools import flatten, transpose


class PerceptronNetwork():
    """Immutable perceptron network (stack of layers)"""

    def __init__(self, layers):
        self._layers = layers
        for idx in range(0, len(layers) - 1):
            if layers[idx].output_size() != layers[idx + 1].input_size():
                raise Exception(
                    """Layer {layer_out} output size of {out_size} mismatch at with
                    layer {layer_in} with input size of {in_size}.""".format(
                        out_size=layers[idx].output_size(),
                        in_size=layers[idx + 1].input_size(),
                        layer_in=layers[idx + 1].name(),
                        layer_out=layers[idx].name()
                    ))

    @staticmethod
    def shorthand(sizes, activation=None):
        """Simple array based init of network"""
        if len(sizes) < 2:
            raise Exception("Invalid size set")
        activation = activation if isinstance(activation, str) else 'sigmoid'
        layers = []
        while len(sizes) > 1:
            input_size = sizes[0]
            sizes = sizes[1:]
            layers.append(PerceptronLayer.blank(input_size,
                                                sizes[0],
                                                str(len(sizes)),
                                                ['?'] * input_size,
                                                activation,
                                                len(sizes)))
        return PerceptronNetwork(layers)

    def layers(self):
        """Current copy of the layers"""
        return self._layers[:]

    def shape(self):
        """List of tuples of self shape"""
        return [(layer.input_size(), layer.output_size()) for layer in self._layers]

    def forward(self, inputs):
        """Run forward in the network. Returns (all_outputs, final_output)"""
        current_data = inputs[:]
        layer_states = [inputs[:]] * (len(self._layers) + 1)
        # corresponds to inputs that pend for each layer
        # as in, 0 is applied to layer 0, 1 is the output of layer 0, applied to layer 1
        # if the network has 3 layers, then layer state 3 (4th) is the network
        # output

        for idx in range(0, len(self._layers)):
            layer = self._layers[idx]
            current_data = layer.forward(current_data)
            layer_states[idx + 1] = current_data[:]

        return (layer_states[-1], layer_states)

    def forward_verbose(self, inputs):
        """Run forward in the network. Returns (all_outputs, final_output, notes)"""
        current_data = inputs[:]
        layer_states = [inputs[:]] * (len(self._layers) + 1)
        verbosity = []

        for idx in range(0, len(self._layers)):
            layer = self._layers[idx]
            current_data, notes = layer.forward_verbose(current_data)
            layer_states[idx + 1] = current_data[:]
            verbosity = verbosity + notes

        notes = ['Network Forward Pass'] + \
            ['| {}'.format(line) for line in verbosity]
        return (layer_states[-1], layer_states, notes)

    def backward(self, layer_states, truths):
        """Perform backprop in network - gathers unit errors for each layer"""
        return self.backward_verbose(layer_states, truths)[0]

    def backward_verbose(self, layer_states, truths):
        """Perform backprop in network - gathers unit errors for each layer, verbose"""
        # special for last layer - defies standard
        backwards_idx = range(len(self._layers) - 2, -1, -1)

        unit_errors = [None] * len(self._layers)
        unit_errors_layer_final, notes_layer_final = \
            self._layers[-1].backward_last_verbose(layer_states[-1], truths)
        unit_errors[-1] = unit_errors_layer_final
        notes = []

        for idx in backwards_idx:
            # use current error terms to update the current layer
            layer_inputs = layer_states[idx + 1]
            layer_back = self._layers[idx]  # target of backprop
            # layer associated with error_terms
            layer_error = self._layers[idx + 1]  # source of error
            layer_error_unit_errors = unit_errors[idx + 1]

            layer_error_weights = transpose(
                [perceptron.weights()[1:] for perceptron in layer_error.perceptrons()])

            unit_errors[idx], layer_back_notes = layer_back.backward_verbose(
                layer_inputs, layer_error_weights, layer_error_unit_errors, layer_error.names())
            notes = notes + layer_back_notes

        notes_all = ['Network Backpass (Unit Errors)'] + \
            ['| {}'.format(line) for line in notes + notes_layer_final]

        return unit_errors, notes_all

    def update_weights(self, layer_states, unit_errors, learning_rate):
        """A new network with updated weights"""
        inputs = layer_states[:-1]
        layers_new = [layer.update_weights(state, unit_error, learning_rate) for
                      layer, state, unit_error in zip(
                          self._layers, inputs, unit_errors)]
        return PerceptronNetwork(layers_new)

    def update_weights_verbose(self, layer_states, unit_errors, learning_rate):
        """A new network with updated weights"""
        inputs = layer_states[:-1]
        layers_new = [layer.update_weights(state, unit_error, learning_rate) for
                      layer, state, unit_error in zip(
                          self._layers, inputs, unit_errors)]
        layers_notes = [layer.update_weights_verbose(state, unit_error, learning_rate)[1] for
                        layer, state, unit_error in zip(
                            self._layers, inputs, unit_errors)]
        notes = ['Network Update Weights'] + \
            ['| {}'.format(line) for line in flatten(layers_notes)]
        return PerceptronNetwork(layers_new), notes

    def step(self, inputs, outputs, learning_rate):
        """Verbose results from a learning forward/backward step"""

        # Step 1: forward pass - predict
        estimated_results, layer_states, notes_forward = self.forward_verbose(
            inputs)

        # Step 2: back pass - collect errors
        unit_errors, notes_backward = self.backward_verbose(
            layer_states, outputs)

        # Step 3: update weights
        network_updated, notes_weights = self.update_weights_verbose(
            layer_states, unit_errors, learning_rate)

        return estimated_results, network_updated, notes_forward + notes_backward + notes_weights

    def train(self, values_input, values_outputs, learning_rate, epochs, epoch_reporting):
        """Train network over data"""
        network_current = self
        mses = []

        for epoch in range(0, epochs):
            standard_error = []
            for values, results in zip(values_input, values_outputs):
                # Step 1: forward pass - predict
                estimated_results, layer_states = network_current.forward(
                    values)

                # Step 1a: note success rate
                weighted_errors = [result - estimated_result for result,
                                   estimated_result in zip(results, estimated_results)]
                weighted_error = sum(weighted_errors) / len(weighted_errors)
                standard_error.append(weighted_error ** 2)

                # Step 2: back pass - collect errors
                unit_errors = network_current.backward(layer_states, results)

                # Step 3: update weights
                network_current = network_current.update_weights(
                    layer_states, unit_errors, learning_rate)

            if len(mses) < 1:
                mse = sum(standard_error) / len(standard_error)
                mse_first = mse

            if epoch % epoch_reporting == 0:
                mse = sum(standard_error) / len(standard_error)
                mses.append((epoch, mse))

        return network_current, mse, mse_first, mses
