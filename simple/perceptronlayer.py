"""Perceptron Layer"""

import math
from .perceptron import Perceptron
from .tools import flatten


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
    def blank(input_size, output_size, name, input_names=None, activation=None):
        """Produce a Perception Layer based on sizes"""
        input_size_with_bias = input_size + 1
        input_names = input_names if input_names is list else [
            'i{}'.format(i) for i in range(0, input_size_with_bias)]
        activation = activation if isinstance(activation, str) else 'sigmoid'
        weights = [1 / math.sqrt(input_size_with_bias)] * input_size_with_bias

        perceptrons = [Perceptron(weights[:], '{0}.{1}'.format(name, i), input_names, activation)
                       for i in range(0, output_size)]
        return PerceptronLayer(perceptrons, name)

    def forward(self, inputs):
        """Perform feed forward on all Perceptrons"""
        if len(inputs) != self._input_size:
            raise Exception("Incorrect number of inputs")

        return [p.forward(inputs) for p in self._perceptrons]

    def forward_verbose(self, inputs):
        """Perform verbose feed forward on all Perceptrons"""
        if len(inputs) != self._input_size:
            raise Exception("Incorrect number of inputs")

        results = [p.forward(inputs) for p in self._perceptrons]

        notes = [p.forward_verbose(inputs) for p in self._perceptrons]
        verbose = ['Forward pass for Layer {}'.format(self._name)] + \
                  ['> {}'.format(line) for line in flatten(notes)]
        return results, verbose

    def backward(self, outputs, weighted_errors):
        """Find error for each Perceptron based on their output and weighted errors"""
        return [perceptron.backward(output_value, weighted_error) for
                perceptron, output_value, weighted_error in
                zip(self._perceptrons, outputs, weighted_errors)]

    def backward_verbose(self, outputs, weighted_errors):
        """Find error for each Perceptron based on their output and weighted errors, notes"""
        results = [perceptron.backward(output_value, weighted_error) for
                   perceptron, output_value, weighted_error in
                   zip(self._perceptrons, outputs, weighted_errors)]
        notes = [perceptron.backward_verbose(output_value, weighted_error)[1] for
                 perceptron, output_value, weighted_error in
                 zip(self._perceptrons, outputs, weighted_errors)]
        notes = ['Backward Pass for Layer {}'.format(self._name)] + \
            ['> {}'.format(line) for line in flatten(notes)]

        return results, notes

    def update_weights(self, inputs, unit_errors, learning_rate):
        """Updated weights on perceptrons layer"""
        perceptrons_new = [perceptron.update_weights(inputs, unit_error, learning_rate) for
                           perceptron, unit_error in zip(self._perceptrons, unit_errors)]
        return PerceptronLayer(perceptrons_new, self._name)
