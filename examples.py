"""Hard coded NN tests"""

import math
from simple.perceptron import Perceptron
from simple.perceptronlayer import PerceptronLayer
from simple.perceptronnetwork import PerceptronNetwork

INPUTS_DEFAULT = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
LEARNING_RATE = 0.25
EPOCHS_AND = 5000


def line_break():
    """Simple line breaks"""
    print('')
    print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
    print('')


def report_network(data, truths, network):
    """Print function for Network Performance"""
    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(data, truths):
        estimated_value, _ = network.forward(value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result[0], round(estimated_value[0], 3)))


def print_notes(notes):
    """Print the notes in a pleasing format"""
    for note in notes:
        print(note)


def try_network(title, inputs, outputs, network, epochs):
    """Run a network against a data set"""
    line_break()
    print('Network {}'.format(title))
    new_network, mse, mse_first, mses = network.train(
        inputs, outputs, LEARNING_RATE, epochs, math.floor(epochs / 5))
    for epoch, mse in mses:
        print('For epoch {0}, MSE of {1}'.format(epoch, mse))
    print('From MSE of {} to {}'.format(mse_first, mse))

    report_network(inputs, outputs, new_network)
    return new_network


def try_network_verbose(title, inputs, outputs, network, epochs):
    """Run a network against a data set - detailing each step"""
    line_break()
    print('Network {} Verbose'.format(title))
    epochs = epochs if isinstance(epochs, int) and epochs > 0 else 2

    for epoch in range(0, epochs):
        for value, results in zip(inputs, outputs):
            print(' ================================================== ')
            print('Epoch {} : {} -> {}'.format(epoch, value, results))
            print(' ================================================== ')

            _, network, notes = network.step(value, results, LEARNING_RATE)
            print_notes(notes)


def try_and():
    """Run a single perceptron with AND logic"""
    line_break()
    print(' Simple single Perceptron AND')
    outputs = [1 if a + b == 2 else 0 for a, b in INPUTS_DEFAULT]
    perceptron = Perceptron([0.5, 0.5, 0.5], 'c', ['a', 'b'])

    for epoch in range(0, EPOCHS_AND):
        standard_error = []
        for value, result in zip(INPUTS_DEFAULT, outputs):
            # Step 1: forward pass - predict
            estimated_value = perceptron.forward(value)

            # Step 2: back pass - collect errors
            weighted_error = result - estimated_value
            standard_error.append(weighted_error ** 2)
            unit_error = perceptron.backward(estimated_value, weighted_error)

            # Step 3: update weights
            perceptron = perceptron.update_weights(
                value, unit_error, LEARNING_RATE)

        if epoch % 1000 == 0:
            print('For epoch {0}, MSE of {1}'.format(
                epoch, sum(standard_error) / len(standard_error)))

    print('Final MSE {0}'.format(sum(standard_error) / len(standard_error)))
    print(' {0:>6} | {1:>5} {2:<10}'.format('Value', 'Truth', 'Prediction'))
    for value, result in zip(INPUTS_DEFAULT, outputs):
        estimated_value = perceptron.forward(value)
        print(' {0:>6} | {1:>5} {2:<10}'.format(
            str(value), result, round(estimated_value, 3)))


def try_and_network():
    """Run an example with Networked AND logic"""
    network = PerceptronNetwork.shorthand([2, 1])
    outputs = [[1] if a + b == 2 else [0] for a, b in INPUTS_DEFAULT]
    return try_network('AND', INPUTS_DEFAULT, outputs, network, 5000)


def try_nor():
    """Run an example with NOR logic"""
    network = PerceptronNetwork.shorthand([2, 1], 'relu')
    outputs = [[1] if a + b == 0 else [0] for a, b in INPUTS_DEFAULT]
    return try_network('NOR', INPUTS_DEFAULT, outputs, network, 5000)


def try_multi():
    """Run an example with junk logic"""
    network = PerceptronNetwork.shorthand([3, 4, 5, 4, 3, 1], 'relu')
    inputs = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]
    outputs = [[1] if a == 0 and (b == 1 or c == 0) else [0]
               for a, b, c in inputs]
    return try_network('!A & (B | !C)', inputs, outputs, network, 5000)


def try_xor():
    """Run an example with XOR logic"""
    network = PerceptronNetwork.shorthand([2, 3, 1], 'relu')
    print(network.shape())
    outputs = [[1] if a + b == 1 else [0] for a, b in INPUTS_DEFAULT]
    return try_network('XOR', INPUTS_DEFAULT, outputs, network, 5000)


def try_xor_verbose():
    """Run an example with XOR logic"""
    line_break()
    print('Network with XOR Verbose')
    network = PerceptronNetwork(
        [
            PerceptronLayer([
                Perceptron(Perceptron.generate_weights(
                    2, 3), 'A', ['X', 'Y'], 'sigmoid'),
                Perceptron(Perceptron.generate_weights(
                    2, 5), 'B', ['X', 'Y'], 'sigmoid'),
                Perceptron(Perceptron.generate_weights(
                    2, 40), 'C', ['X', 'Y'], 'sigmoid')
            ], 'main'),
            PerceptronLayer([
                Perceptron(Perceptron.generate_weights(3, 9),
                           'D', ['A', 'B', 'C'], 'sigmoid')
            ], 'final')
        ]
    )
    outputs = [[1] if a + b == 1 else [0] for a, b in INPUTS_DEFAULT]
    return try_network_verbose('XOR', INPUTS_DEFAULT, outputs, network, 2)


def try_class_example_verbose():
    """Run an example from class, verbosely"""
    network = PerceptronNetwork(
        [
            PerceptronLayer([
                Perceptron([0.1] * 3, 'C', ['A', 'B'], 'sigmoid')
            ], 'main'),
            PerceptronLayer([
                Perceptron([0.1] * 2, 'D', ['C'], 'sigmoid')
            ], 'final')
        ]
    )
    inputs = [[1, 0], [0, 1]]
    outputs = [[1], [0]]
    return try_network_verbose('Class Example', inputs, outputs, network, 2)


if __name__ == "__main__":
    try_and()
    try_and_network()
    try_nor()
    try_xor()
    try_multi()

    try_xor_verbose()
    try_class_example_verbose()
