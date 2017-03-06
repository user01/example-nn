"""Tools for Neural Network built by hand"""

import math


def linear_forward(inputs, weights):
    """netj; Sum inputs and weight products"""
    return linear_forward_details(inputs, weights)['full_value']

def linear_forward_details(inputs, weights):
    """Detailed Sum inputs and weight products"""
    if len(inputs) + 1 != len(weights):
        raise Exception("Inputs and Weights size mismatch")
    inputs_with_bias = [1] + inputs
    inputs_and_weights = zip(inputs_with_bias, weights)
    inputs_by_weights = [p[0] * p[1] for p in inputs_and_weights]
    full_value = sum(inputs_by_weights)
    return {
        'inputs_with_bias': inputs_with_bias,
        'inputs_and_weights': inputs_and_weights,
        'inputs_by_weights': inputs_by_weights,
        'full_value': full_value
    }


def linear_backward(inputs, error, weights, learning_rate):
    """New weights. Updated based on inputs, weights, node error, and learning rate"""
    return linear_backward_details(inputs, error, weights, learning_rate)['weights_updated']

def linear_backward_details(inputs, error, weights, learning_rate):
    """Detailed values based on inputs, weights, node error, and learning rate"""
    inputs_with_bias = [1] + inputs
    weights_delta = [learning_rate * error * i for i in inputs_with_bias]
    weights_updated = [p[0] + p[1] for p in zip(weights, weights_delta)]
    return {
        'inputs': inputs,
        'learning_rate': learning_rate,
        'error': error,
        'weights_delta': weights_delta,
        'weights': weights,
        'weights_updated': weights_updated
    }


def sigmoid_forward(value):
    """Sigmoid function"""
    return 1 / (1 + math.exp(value))


def sigmoid_backward(value):
    """Backwards Sigmoid function (gradient)"""
    return value * (1 - value)
