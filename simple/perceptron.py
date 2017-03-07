
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

    def input_counts(self):
        """Returns a copy of the current weights"""
        return len(self._weights) - 1

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
        self._input_size = perceptrons[0].input_counts()
        self._name = name
        self._perceptrons = perceptrons

    def perceptrons(self):
        """Copy of layer perceptrons"""
        return self._perceptrons[:]

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

