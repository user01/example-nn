"""Perceptron Layer"""

import math
from .perceptron import Perceptron
from .tools import flatten, float_fix, intersperse, align_equations


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

    def names(self):
        """Names of all perceptrons"""
        return [perceptron.name() for perceptron in self._perceptrons]

    @staticmethod
    def blank(input_size, output_size, name, input_names=None, activation=None):
        """Produce a Perception Layer based on sizes"""
        input_size_with_bias = input_size + 1
        input_names = input_names if input_names is list else [
            'i{}'.format(i) for i in range(0, input_size_with_bias)]
        activation = activation if isinstance(activation, str) else 'sigmoid'

        perceptrons = [Perceptron(
            Perceptron.generate_weights(
                input_size, i + input_size + output_size),
            '{0}.{1}'.format(name, i), input_names, activation) for i in range(0, output_size)]
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
        verbose = ['Forward pass for Layer "{}"'.format(self._name)] + \
                  ['> {}'.format(line) for line in flatten(notes)]
        return results, verbose

    def backward_last(self, output_units, truths):
        """Error for each Perceptron ONLY when it's the last layer"""
        return self.backward_last_verbose(output_units, truths)[0]

    def backward_last_verbose(self,
                              output_units,
                              truths):
        """Find error for each Perceptron ONLY when it's the last layer, verbose"""

        details = [PerceptronLayer._backward_last_details(perceptron, output_unit, truth) for
                   perceptron, output_unit, truth in
                   zip(self._perceptrons, output_units, truths)]

        results = [detail[0] for detail in details]
        lines = flatten([detail[1] for detail in details])

        notes = ['Backward Pass for Final Layer "{}"'.format(self._name)] + \
            ['> {}'.format(line) for line in align_equations(lines)]

        return results, notes

    def backward(self, output_units, weights_for_unit, output_errors):
        """Find error for each Perceptron based on their output and weighted errors"""
        return self.backward_verbose(output_units,
                                     weights_for_unit,
                                     output_errors,
                                     [''] * len(output_errors))

    def backward_verbose(self,
                         # List<output> for each of this layer's perceptrons
                         output_units,
                         # List<List<weight>>, each List<weight> feed from idx
                         # perceptron
                         weights_for_unit,
                         # List<error> errors associated with the next layer
                         # (each output)
                         output_errors,
                         output_names):
        """Find error for each Perceptron"""

        details = [PerceptronLayer._backward_single_details(perceptron,
                                                            output_unit,
                                                            weights,
                                                            output_errors,
                                                            output_names) for
                   perceptron, output_unit, weights in
                   zip(self._perceptrons, output_units, weights_for_unit)]

        results = [detail[0] for detail in details]
        lines = flatten([detail[1] for detail in details])

        notes = ['Backward Pass for Layer "{}"'.format(self._name)] + \
            ['> {}'.format(line) for line in align_equations(lines)]

        return results, notes

    @staticmethod
    def _backward_single_details(perceptron, output_unit, output_weights,
                                 output_errors, output_names):
        """Verbose results for backwards of a perceptron (compute unit error)"""
        name = perceptron.name()
        weighted_error = sum([weight * error for weight,
                              error in zip(output_weights, output_errors)])

        # weighted_errors_labels = [weight, error, weight * error for weight,
        # error in zip(output_weights, output_errors)]
        weights_errors_text = flatten(
            intersperse([
                ['w_{}.{}'.format(name, output_name), '*', '𝛿_{}'.format(output_name)] for
                output_name in output_names], ['+']))

        weights_errors_values_1 = flatten(
            intersperse([
                [float_fix(weight), '*', float_fix(error)] for weight,
                error in zip(output_weights, output_errors)], ['+']))

        weights_errors_values_2 = flatten(
            intersperse([
                ['', '', float_fix(weight * error)] for weight,
                error in zip(output_weights, output_errors)], ['+']))

        details = perceptron.backward_details(output_unit, weighted_error)

        notes = [
            [
                "unit_error({})".format(name), "=",
                "σ'_{}(out_{})".format(perceptron.activation(), name), "*",
                "["
            ] + weights_errors_text + ["]"],
            [
                "unit_error({})".format(name), "=",
                "σ'_{}(out_{})".format(perceptron.activation(), name), "*",
                "["
            ] + weights_errors_values_1 + ["]"],
            [
                "unit_error({})".format(name), "=",
                "σ'_{}(out_{})".format(perceptron.activation(), name), "*",
                "["
            ] + weights_errors_values_2 + ["]"],
            [
                "unit_error({})".format(name), "=",
                float_fix(details['activation_backwards']), "*",
                "[", float_fix(details['weighted_error']), "]",
                "=", float_fix(details["unit_error"])
            ]
        ]

        return details["unit_error"], notes

    @staticmethod
    def _backward_last_details(perceptron, output_unit, truth):
        """Verbose results for backwards of a perceptron (compute unit error)"""
        name = perceptron.name()
        weighted_error = truth - output_unit
        details = perceptron.backward_details(output_unit, weighted_error)

        notes = [
            [
                "unit_error({})".format(name), "=",
                "σ'_{}(out_{})".format(perceptron.activation(), name), "*",
                "[", "Truth", "-",
                "out_{}".format(name), "]"
            ],
            [
                "unit_error({})".format(name), "=",
                "σ'_{}({})".format(perceptron.activation(), float_fix(output_unit)), "*",
                "[", float_fix(truth), "-",
                float_fix(output_unit), "]"
            ]
        ]

        return details["unit_error"], notes

    def update_weights(self, inputs, unit_errors, learning_rate):
        """Updated weights on perceptrons layer"""
        perceptrons_new = [perceptron.update_weights(inputs, unit_error, learning_rate) for
                           perceptron, unit_error in zip(self._perceptrons, unit_errors)]
        return PerceptronLayer(perceptrons_new, self._name)

    def update_weights_verbose(self, inputs, unit_errors, learning_rate):
        """Updated weights on perceptrons layer"""
        notes = ['Updated weights for Layer "{}"'.format(self._name)]
        perceptrons_new = [perceptron.update_weights(inputs, unit_error, learning_rate) for
                           perceptron, unit_error in zip(self._perceptrons, unit_errors)]
        perceptrons_notes = [perceptron.update_weights_verbose(inputs, unit_error,
                                                               learning_rate)[1] for
                             perceptron, unit_error in zip(self._perceptrons, unit_errors)]

        return PerceptronLayer(perceptrons_new, self._name), notes + \
            ['> {}'.format(line) for line in flatten(perceptrons_notes)]
