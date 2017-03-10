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
    fix_num = lambda value: str(round(value, 3))
    results = linear_forward_details(inputs, weights)
    inputs_with_bias = [1] + inputs
    input_names_with_bias = ['BIAS'] + input_names
    header = ['net_{}'.format(name), '=']
    tailer = [fix_num(results['netj']), '=']

    lines = header + flatten([['w_{name}.{source}'.format(name=name, source=source), '*',
                               'x_{name}.{source}'.format(name=name, source=source), '+']
                              for source in input_names_with_bias])[:-1]

    numbers = tailer + flatten([[fix_num(weight), '*', fix_num(input_value), '+']
                                for weight, input_value in zip(weights, inputs_with_bias)])[:-1]

    return align_equations([lines, numbers])


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


def linear_backward_verbose(inputs, name, input_names, unit_error, weights, learning_rate):
    """List of strings detailing the backwards step"""
    details = linear_backward_details(
        inputs, unit_error, weights, learning_rate)
    inputs_with_bias = [1] + inputs
    input_names_with_bias = ['BIAS'] + input_names
    weights_updated = details['weights_updated']
    weights = details['weights']
    weights_delta = details['weights_delta']

    lines = [
        ['w_{name}.{source}'.format(name=name, source=source), '<-',
         'w_{name}.{source}'.format(name=name, source=source), '+',
         'Δw_{name}.{source}'.format(name=name, source=source), '=',
         'w_{name}.{source}'.format(name=name, source=source), '+',
         'η', '*',
         '𝛿_{name}'.format(name=name), '*',
         'x_{name}.{source}'.format(name=name, source=source)]
        for source in input_names_with_bias]

    fix_num = lambda value: str(round(value, 3))

    numbers = [
        [fix_num(weight_updated), '<-',
         fix_num(weight), '+',
         fix_num(weight_delta), '=',
         fix_num(weight), '+',
         fix_num(learning_rate), '*',
         fix_num(unit_error), '*',
         fix_num(input_value)]
        for weight_updated, weight, weight_delta, input_value in
        zip(weights_updated, weights, weights_delta, inputs_with_bias)]

    return align_equations(interleave(lines, numbers))


def sigmoid_forward(value):
    """Sigmoid function"""
    return 1 / (1 + math.exp(-value))


def sigmoid_backward(value):
    """Backwards Sigmoid function (gradient)"""
    return value * (1 - value)


def relu_forward(value):
    """Relu function"""
    return max(0, value)


def relu_backward(value):
    """Backwards ReLu function (gradient)"""
    return 1 if value > 0 else 0


def tanh_forward(value):
    """tanh function"""
    return (math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value))


def tanh_backward(value):
    """Backwards tanh function (gradient)"""
    return 1 - math.tanh(value) ** 2


# http://stackoverflow.com/a/11125298/2601448
def interleave(list_a, list_b):
    """Interleaves two lists of the same length"""
    return [x for t in zip(list_a, list_b) for x in t]


def intersperse(list_main, element):
    """Intersperse an element between each element in the main list."""
    list_element = [element] * len(list_main)
    list_all = interleave(list_main, list_element)
    return list_all[:-1]


def flatten(lst):
    """Flatten a List<List<any>> into List<any>"""
    # http://stackoverflow.com/a/952952/2601448
    return [item for sublist in lst for item in sublist]


def float_fix(value):
    """Returns a readable string of a float"""
    return str(round(value, 3))


def transpose(lst):
    """Transpose a List<List<any>> structure"""
    return [list(elm) for elm in zip(*lst)]


def align_equations(equations):
    """Convert n x m lists into aligned strings"""
    max_equation_size = 0
    for equation in equations:
        max_equation_size = max(len(equation), max_equation_size)
    equations = [equation + [''] *
                 (max_equation_size - len(equation)) for equation in equations]

    nth_size = [None] * len(equations[0])
    for idx in range(0, len(equations[0])):
        nth_size[idx] = max([len(equation[idx]) for equation in equations])

    new_equations = [None] * len(equations)
    for idx in range(0, len(equations)):
        new_equations[idx] = ' '.join([('{0:>' + str(size) + '}').format(term)
                                       for term, size in zip(equations[idx], nth_size)])

    return new_equations
