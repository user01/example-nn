
"""Perceptrons"""

from .tools import linear_forward, sigmoid_forward, relu_forward, tanh_forward
from .tools import linear_backward, sigmoid_backward, relu_backward, tanh_backward
from .tools import linear_forward_verbose


class Perceptron():
    """Immutable Perceptron unit"""

    def __init__(self, weights, name, input_names, activation=None):
        self._weights = weights
        self._name = name
        self._input_names = input_names
        self._activation = Perceptron._activation_string(activation)

    @staticmethod
    def _activation_string(value):
        """Convert string to known activation function string"""
        value = 'sigmoid' if value is not str else value
        value = value.lower()
        if value == 'relu' or value == 'tanh':
            return value
        return 'sigmoid'

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
        output_j = self.activation_forward(net_j)
        return output_j

    def activation_forward(self, value):
        """Perform activation on the perceptron"""
        if self._activation == 'relu':
            return relu_forward(value)
        if self._activation == 'tanh':
            return tanh_forward(value)
        return sigmoid_forward(value)

    def activation_backward(self, value):
        """Perform activation on the perceptron"""
        if self._activation == 'relu':
            return relu_backward(value)
        if self._activation == 'tanh':
            return tanh_backward(value)
        return sigmoid_backward(value)

    def forward_verbose(self, inputs):
        """Verbose results from the forward pass"""
        linear_prints = linear_forward_verbose(
            inputs, self._weights, self._name, self._input_names)
        net_j = linear_forward(inputs, self._weights)
        output_j = self.activation_forward(net_j)
        sigmoid_prints = [
            ' σ(net_{0}) = σ({1}) = {2}'.format(
                self._name, round(net_j, 3), round(output_j))
        ]
        return linear_prints + sigmoid_prints

    def backward(self, output_unit, weighted_error):
        """Returns updated weights"""
        unit_error = self.activation_backward(output_unit) * weighted_error
        return unit_error

    def update_weights(self, inputs, unit_error, learning_rate):
        """Returns updated Perceptron"""
        weights_updated = linear_backward(
            inputs, unit_error, self._weights, learning_rate)
        return Perceptron(weights_updated, self._name, self._input_names)
