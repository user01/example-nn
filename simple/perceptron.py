
"""Perceptions"""

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
        """Returns updated weights and """
        unit_error = sigmoid_backward(output_unit) * weighted_error
        return unit_error

    def update_weights(self, inputs, unit_error, learning_rate):
        """Returns updated Perceptron"""
        weights_updated = linear_backward(
            inputs, unit_error, self._weights, learning_rate)
        return Perceptron(weights_updated, self._name, self._input_names)
