"""Tools for Neural Network built by hand"""

import math

def forward_pass_sum(inputs, weights):
    """Sum inputs and weight products"""
    if len(inputs) + 1 != len(weights):
        raise Exception("Inputs and Weights size mismatch")
    inputs_with_bias = [1] + inputs
    inputs_and_weights = zip(inputs_with_bias, weights)
    inputs_by_weights = [p[0] * p[1] for p in inputs_and_weights]
    full_value = sum(inputs_by_weights)
    return full_value

def sigmoid(value):
    """Sigmoid function"""
    return 1 / (1 + math.exp(value))
