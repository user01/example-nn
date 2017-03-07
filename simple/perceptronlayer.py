"""Perceptron Layer"""

import math
from .perceptron import Perceptron


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
        print('Starting weights', weights)

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