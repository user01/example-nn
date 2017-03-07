
"""Perceptions"""

import math

from .tools import linear_forward, sigmoid_forward
from .tools import linear_backward, sigmoid_backward
from .tools import linear_forward_verbose


class Perceptron():
    """Immutable Perceptron unit"""

    def __init__(self, weights, name, input_names):
        self._weights = weights
        self._name = name
        self._input_names = input_names

    def weights(self):
        """Returns a copy of the current weights"""
        return self._weights[:]

    def input_size(self):
        """Returns the size of the input"""
        return len(self._weights) - 1

    def name(self):
        """Returns the perceptron name."""
        return self._name

    def forward(self, inputs):
        """Run an input through the Perceptron. Returns the output value"""
        net_j = linear_forward(inputs, self._weights)
        output_j = sigmoid_forward(net_j)
        return output_j

    def forward_verbose(self, inputs):
        """Verbose results from the forward pass"""
        linear_prints = linear_forward_verbose(
            inputs, self._weights, self._name, self._input_names)
        net_j = linear_forward(inputs, self._weights)
        output_j = sigmoid_forward(net_j)
        sigmoid_prints = [
            ' σ(net_{0}) = σ({1}) = {2}'.format(
                self._name, round(net_j, 3), round(output_j))
        ]
        return linear_prints + sigmoid_prints

    def backward(self, output_unit, weighted_error):
        """Returns updated weights"""
        unit_error = sigmoid_backward(output_unit) * weighted_error
        return unit_error

    def update_weights(self, inputs, unit_error, learning_rate):
        """Returns updated Perceptron"""
        weights_updated = linear_backward(
            inputs, unit_error, self._weights, learning_rate)
        return Perceptron(weights_updated, self._name, self._input_names)


class PerceptronLayer():
    """Container for multiple Perceptrons in a layer"""

    def __init__(self, perceptrons, name):
        self._input_size = perceptrons[0].input_size()
        self._output_size = len(perceptrons)
        self._name = name
        self._perceptrons = perceptrons

    def perceptrons(self):
        """Copy of layer perceptrons"""
        return self._perceptrons[:]

    def input_size(self):
        """Size of input array"""
        return self._input_size

    def output_size(self):
        """Size out output array"""
        return self._output_size

    def name(self):
        """Name of Layer"""
        return self._name

    @staticmethod
    def blank(input_size, output_size, name, input_names):
        """Produce a Perception Layer based on sizes"""
        input_size_with_bias = input_size + 1
        weights = [1 / math.sqrt(input_size_with_bias)] * input_size_with_bias

        perceptrons = [Perceptron(weights[:], '{0}.{1}'.format(name, i), input_names)
                       for i in range(0, output_size)]
        return PerceptronLayer(perceptrons, name)

    def forward(self, inputs):
        """Perform feed forward on all Perceptrons"""
        if len(inputs) != self._input_size:
            raise Exception("Incorrect number of inputs")

        return [p.forward(inputs) for p in self._perceptrons]

    def backward(self, outputs, weighted_errors):
        """Find error for each Perceptron based on their output and weighted errors"""
        return [perceptron.backward(output_value, weighted_error) for
                perceptron, output_value, weighted_error in
                zip(self._perceptrons, outputs, weighted_errors)]

    def update_weights(self, inputs, unit_errors, learning_rate):
        """Updated weights on perceptrons layer"""
        perceptrons_new = [perceptron.update_weights(inputs, unit_error, learning_rate) for
                           perceptron, unit_error in zip(self._perceptrons, unit_errors)]
        return PerceptronLayer(perceptrons_new, self._name)


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

    def backward(self, layer_states, truths):
        """Perform backprop in network - gathers unit errors for each layer"""
        # new_layers = [None] * len(self._layers)
        # skip last layer - defies standard
        backwards_idx = range(len(self._layers) - 2, -1, -1)
        error_terms = [truth - output for truth,
                       output in zip(truths, layer_states[-1])]

        unit_errors = [None] * len(self._layers)
        unit_errors_layer_final = self._layers[
            -1].backward(layer_states[-1], error_terms)
        unit_errors[-1] = unit_errors_layer_final

        for idx in backwards_idx:
            # use current error terms to update the current layer
            layer_inputs = layer_states[idx + 1]
            layer_back = self._layers[idx]
            # layer associated with error_terms
            layer_error = self._layers[idx + 1]
            layer_error_unit_errors = unit_errors[idx + 1]

            perceptron_count = len(layer_back.perceptrons())
            weighted_errors = [None] * perceptron_count
            for i in range(0, perceptron_count):
                # for every node in the error layer, sum( (i+1)th weight * node_unit_error )
                # is the weighted_error for the ith node in layer_back
                weighted_errors[i] = sum([perceptron.weights()[i + 1] * unit_error for
                                          perceptron, unit_error in zip(
                                              layer_error.perceptrons(), layer_error_unit_errors)])

            unit_errors[idx] = layer_back.backward(
                layer_inputs, weighted_errors)

        return unit_errors

    def update_weights(self, layer_states, unit_errors, learning_rate):
        """A new network with updated weights"""
        inputs = layer_states[:-1]
        layers_new = [layer.update_weights(state, unit_error, learning_rate) for
                      layer, state, unit_error in zip(
                          self._layers, inputs, unit_errors)]
        return PerceptronNetwork(layers_new)

    def train(self, values_input, values_outputs, learning_rate, epochs, epoch_reporting):
        """Train network over data"""
        network_current = self
        mses = []

        for epoch in range(0, epochs):
            # print('Starting Epoch {}'.format(epoch))
            standard_error = []
            for values, results in zip(values_input, values_outputs):
                # print('For {0}, truth {1} ...'.format(value, result))

                # Step 1: forward pass - predict
                estimated_results, layer_states = network_current.forward(
                    values)

                # Step 2: back pass - collect errors
                weighted_errors = [result - estimated_result for result,
                                   estimated_result in zip(results, estimated_results)]
                weighted_error = sum(weighted_errors) / len(weighted_errors)
                standard_error.append(weighted_error ** 2)
                unit_errors = network_current.backward(
                    layer_states, weighted_errors)

                # Step 3: update weights
                network_current = network_current.update_weights(
                    layer_states, unit_errors, learning_rate)

            if len(mses) < 1:
                mse = sum(standard_error) / len(standard_error)
                mse_first = mse

            if epoch % epoch_reporting == 0:
                mse = sum(standard_error) / len(standard_error)
                mses.append((epoch, mse))
                # print('For epoch {0}, MSE of {1}'.format(epoch, mse))

        return network_current, mse, mse_first, mses
        # print('Final MSE {0}'.format(sum(standard_error) / len(standard_error)))
        # print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
        # for value, result in zip(values_input, values_outputs):
        #     estimated_value = perceptron.forward(value)
        #     # print(value, result, estimated_value)
        #     print(' {0:>6} | {1:>5} {2:<10}'.format(
        #         str(value), result, round(estimated_value, 3)))
