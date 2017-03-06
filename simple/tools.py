"""Tools for Neural Network built by hand"""

import math


def linear_forward(inputs, weights):
    """netj; Sum inputs and weight products"""
    return linear_forward_details(inputs, weights)['netj']


def linear_forward_details(inputs, weights):
    """Detailed Sum inputs and weight products"""
    if len(inputs) + 1 != len(weights):
        raise Exception("Inputs and Weights size mismatch")
    inputs_with_bias = [1] + inputs
    inputs_and_weights = list(zip(inputs_with_bias, weights))
    inputs_by_weights = [p[0] * p[1] for p in inputs_and_weights]
    netj = sum(inputs_by_weights)
    return {
        'inputs_with_bias': inputs_with_bias,
        'inputs_and_weights': inputs_and_weights,
        'inputs_by_weights': inputs_by_weights,
        'netj': netj
    }


def linear_forward_verbose(inputs, weights, name, input_names):
    """List of strings detailing the final and intermediate results of the step"""
    results = linear_forward_details(inputs, weights)
    inputs_with_bias = [1] + inputs
    input_names_with_bias = ['BIAS'] + input_names
    header = 'net_{0} = {1} = '.format(name, round(results['netj'], 3))
    input_names_max = max([len(i) for i in input_names_with_bias]) + 6
    input_line = ' {0:>' + str(input_names_max) + \
        '} * {1:<' + str(input_names_max) + '}'
    lines = [input_line.format('w_' + name + '.' + source, 'x_' + name + '.' + source)
             for source in input_names_with_bias]
    number_line = ' {0:' + str(input_names_max) + \
        '} * {1:<' + str(input_names_max) + '} '
    numbers = [number_line.format(round(w, input_names_max), round(i, input_names_max))
               for (w, i) in zip(weights, inputs_with_bias)]
    return [header] + interleave(lines, numbers)


def linear_backward(inputs, error, weights, learning_rate):
    """New weights. Updated based on inputs, weights, node error, and learning rate"""
    return linear_backward_details(inputs, error, weights, learning_rate)['weights_updated']


def linear_backward_details(inputs, unit_error, weights, learning_rate):
    """Detailed values based on inputs, weights, unit error, and learning rate"""
    inputs_with_bias = [1] + inputs
    weights_delta = [learning_rate * unit_error * i for i in inputs_with_bias]
    weights_updated = [p[0] + p[1] for p in zip(weights, weights_delta)]
    return {
        'inputs': inputs,
        'learning_rate': learning_rate,
        'unit_error': unit_error,
        'weights_delta': weights_delta,
        'weights': weights,
        'weights_updated': weights_updated
    }


def sigmoid_forward(value):
    """Sigmoid function"""
    return 1 / (1 + math.exp(-value))


def sigmoid_backward(value):
    """Backwards Sigmoid function (gradient)"""
    return value * (1 - value)


# http://stackoverflow.com/a/11125298/2601448
def interleave(list_a, list_b):
    """Interleaves two lists of the same length"""
    return [x for t in zip(list_a, list_b) for x in t]
